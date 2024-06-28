##########################################
############  NSParams #####################
##########################################
mutable struct NSParams
    x::Any
    N::Any
    K::Any
    Kf::Any
    k::Any
    ky::Any
    nu::Any
    normk::Any
    f::Any
    Pxx::Any
    Pxy::Any
    Pyy::Any
    prefactor_F::Any
    ∂x::Any
    ∂y::Any
end

# Store parameters and precomputed operators in a named tuple to toss around.
# Having this in a function gets useful when we later work with multiple
# resolutions.
function create_params(
        K;
        nu,
        f = z(2K, 2K),
        anti_alias_factor = 2 / 3
)
    Kf = round(Int, anti_alias_factor * K)
    N = 2K
    x = LinRange(0.0f0, 1.0f0, N + 1)[2:end]

    ## Vector of wavenumbers
    k = ArrayType(fftfreq(N, Float32(N)))
    normk = k .^ 2 .+ k' .^ 2

    ## Projection components
    kx = k
    ky = reshape(k, 1, :)
    Pxx = @. 1 - kx * kx / (kx^2 + ky^2)
    Pxy = @. 0 - kx * ky / (kx^2 + ky^2)
    Pyy = @. 1 - ky * ky / (kx^2 + ky^2)

    ## The zeroth component is currently `0/0 = NaN`. For `CuArray`s,
    ## we need to explicitly allow scalar indexing.
    CUDA.@allowscalar Pxx[1, 1] = 1
    CUDA.@allowscalar Pxy[1, 1] = 0
    CUDA.@allowscalar Pyy[1, 1] = 1

    # Prefactor for F
    pf = Zygote.@ignore nu * (2.0f0π)^2 * normk

    # Partial derivatives
    ∂x = 2.0f0π * im * k
    ∂y = 2.0f0π * im * ky

    NSParams(x, N, K, Kf, k, ky, nu, normk, f, Pxx, Pxy, Pyy, pf, ∂x, ∂y)
end

##########################################
############  Data  ######################
##########################################
# This is a struct that contains the data of LES and DNS
struct Data
    t::Array{Float32, 1}
    # u is usually the DNS solution
    u::Array{Array{ComplexF32, 4}, 1}
    # v is the LES solution (filtered DNS ubar)
    v::Array{ComplexF32, 5}
    # commutator error that can be used for derivative fitting
    c::Array{ComplexF32, 5}
    params_les::NSParams
    params_dns::NSParams
end
# Function to get the name of the file where to store data
function get_data_name(nu::Float32, les_size::Int, dns_size::Int, myseed::Int)
    return "DNS_$(dns_size)_LES_$(les_size)_nu_$(nu)_$(myseed)"
end

##########################################
############  Cache  #####################
##########################################
mutable struct NSCache
    du::Any
    uf::Any
    v::Any
    vx::Any
    vy::Any
    v2::Any
    v2_reshaped::Any
    qf::Any
    q::Any
    F::Any
    Q::Any
end
function create_cache(
        u,
)
    n = size(u, 1)
    du = similar(u)
    uf = similar(u)
    v = similar(u)
    vx, vy = eachslice(v; dims = 3)
    v2 = z(n, n, 4, size(u, 4))
    v2_reshaped = reshape(view(v2, :), n, n, 2, 2, size(u, 4))
    qf = zeros(Complex{MY_TYPE}, size(v2_reshaped))
    q = similar(u)
    F = similar(u)
    Q = similar(u)

    NSCache(du, uf, v, vx, vy, v2, v2_reshaped, qf, q, F, Q)
end

######################################################
############  Initial Condition ######################
######################################################
# For the initial conditions, we create a random spectrum with some decay.
# Note that the initial conditions are projected onto the divergence free
# space at the end.
function create_spectrum(params; A, σ, s)
    T = eltype(params.x)
    kx = params.k
    ky = reshape(params.k, 1, :)
    τ = 2.0f0π
    a = @. A / sqrt(τ^2 * 2σ^2) *
           exp(-(kx - s)^2 / 2σ^2 - (ky - s)^2 / 2σ^2 - im * τ * rand(T))
    a
end
function random_field(params; A = 1.0f6, σ = 30.0f0, s = 5.0f0, nsamp = 1)
    batch_u = [begin
                   ux = create_spectrum(params; A, σ, s)
                   uy = create_spectrum(params; A, σ, s)
                   u = cat(ux, uy; dims = 3)
                   u = real.(ifft(u, (1, 2)))
                   u = fft(u, (1, 2))
                   ux, uy = eachslice(u; dims = 3)
                   dux = @. params.Pxx * ux + params.Pxy * uy
                   duy = @. params.Pxy * ux + params.Pyy * uy
                   cat(dux, duy; dims = 3)
               end
               for _ in 1:nsamp]
    cat(batch_u..., dims = 4)
end

##########################################
############  Force ######################
##########################################
# The function `Q` computes the quadratic term.
# The `K - Kf` highest frequencies of `u` are cut-off to prevent aliasing.
function Q(u, params, cache)
    n = size(u, 1)
    Kz = params.K - params.Kf

    ## Remove aliasing components
    copyto!(cache.uf, u)
    @views begin
        @. cache.uf[1:(params.Kf), (params.Kf + 1):(n - params.Kf), :, :] .= 0
        @. cache.uf[(params.Kf + 1):(n - params.Kf), :, :, :] .= 0
        @. cache.uf[(n - params.Kf + 1):n, (params.Kf + 1):(n - params.Kf), :, :] .= 0
    end

    ## Spatial velocity
    cache.v .= real.(ifft(cache.uf, (1, 2)))

    ## Quadractic terms in space
    @views begin
        @. cache.v2[:, :, 1, :] = cache.vx .* cache.vx
        @. cache.v2[:, :, 2, :] = cache.vx .* cache.vy
        @. cache.v2[:, :, 3, :] = cache.vx .* cache.vy
        @. cache.v2[:, :, 4, :] = cache.vy .* cache.vy
    end

    ## Quadractic terms in spectral space
    cache.qf .= fft(cache.v2_reshaped, (1, 2))

    ## Compute partial derivatives in spectral space
    @views @. cache.q = -params.∂x * cache.qf[:, :, :, 1, :] -
                        params.∂y * cache.qf[:, :, :, 2, :]

    ## Zero out high wave-numbers 
    @views begin
        @. cache.q[1:(params.Kf), (params.Kf + 1):(n - params.Kf), :, :] .= 0
        @. cache.q[(params.Kf + 1):(n - params.Kf), :, :, :] .= 0
        @. cache.q[(n - params.Kf + 1):n, (params.Kf + 1):(n - params.Kf), :, :] .= 0
    end

    cache.q
end

# `F` computes the unprojected momentum right hand side $\hat{F}$. It also
# includes the closure term (if any).
function F_NS(u, params, cache)
    q = Q(u, params, cache)
    @. cache.F .= q - params.prefactor_F * u + params.f
    cache.F
end

function project(u, params, cache)
    ux, uy = eachslice(u; dims = 3)
    @views begin
        @. cache.du[:, :, 1, :] .= params.Pxx * ux + params.Pxy * uy
        @. cache.du[:, :, 2, :] .= params.Pxy * ux + params.Pyy * uy
    end
    cache.du
end

##########################################
############  Utils ######################
##########################################
# For plotting, the spatial vorticity can be useful. It is given by
# $$
# \omega = -\frac{\partial u_x}{\partial y} + \frac{\partial u_y}{\partial x},
# $$
# which becomes
# $$
# \hat{\omega} = 2 \pi \mathrm{i} k \times u = - 2 \pi \mathrm{i} k_y u_x + 2 \pi \mathrm{i} k_x u_y
# $$
# in spectral space.
function vorticity(u, params)
    ∂x = 2.0f0π * im * params.k
    ∂y = 2.0f0π * im * reshape(params.k, 1, :)
    ux, uy = eachslice(u; dims = 3)
    ω = @. -∂y * ux + ∂x * uy
    real.(ifft(ω))
end
# Function to chop off frequencies and multiply with scaling factor
function spectral_cutoff(u, K)
    dims = ndims(u)
    scaling_factor = MY_TYPE((2K)^2 / (size(u, 1) * size(u, 2)))

    if dims == 3
        result = [u[1:K, 1:K, :] u[1:K, (end - K + 1):end, :]
                  u[(end - K + 1):end, 1:K, :] u[(end - K + 1):end, (end - K + 1):end, :]]
    elseif dims == 4
        result = [u[1:K, 1:K, :, :] u[1:K, (end - K + 1):end, :, :]
                  u[(end - K + 1):end, 1:K, :, :] u[(end - K + 1):end, (end - K + 1):end, :, :]]
    else
        error("Unsupported dimensionality: $dims. `u` must be 3 or 4 dimensions.")
    end

    return scaling_factor * result
end
