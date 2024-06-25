# Function to get the name of the file where to store data
function get_data_name(nu::Float32, les_size::Int, dns_size::Int, myseed::Int)
    return "DNS_$(dns_size)_LES_$(les_size)_nu_$(nu)_$(myseed)"
end

##########################################
############  Params #####################
##########################################
mutable struct Params
    x::Any
    N::Any
    K::Any
    Kf::Any
    k::Any
    nu::Any
    normk::Any
    f::Any
    Pxx::Any
    Pxy::Any
    Pyy::Any
    prefactor_F::Any
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

    Params(x, N, K, Kf, k, nu, normk, f, Pxx, Pxy, Pyy, pf)
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
function random_field(params; A = 1.0f6, σ = 30.0f0, s = 5.0f0)
    ux = create_spectrum(params; A, σ, s)
    uy = create_spectrum(params; A, σ, s)
    u = cat(ux, uy; dims = 3)
    u = real.(ifft(u, (1, 2)))
    u = fft(u, (1, 2))
    project(u, params)
end

##########################################
############  Force ######################
##########################################
# The function `Q` computes the quadratic term.
# The `K - Kf` highest frequencies of `u` are cut-off to prevent aliasing.
function Q(u, params)
    n = size(u, 1)
    Kz = params.K - params.Kf

    ## Remove aliasing components
    uf = [u[1:(params.Kf), 1:(params.Kf), :] z(params.Kf, 2Kz, 2) u[1:(params.Kf), (end - params.Kf + 1):end, :]
          z(2Kz, params.Kf, 2) z(2Kz, 2Kz, 2) z(2Kz, params.Kf, 2)
          u[(end - params.Kf + 1):end, 1:(params.Kf), :] z(params.Kf, 2Kz, 2) u[(end - params.Kf + 1):end, (end - params.Kf + 1):end, :]]

    ## Spatial velocity
    v = real.(ifft(uf, (1, 2)))
    vx, vy = eachslice(v; dims = 3)

    ## Quadractic terms in space
    vxx = vx .* vx
    vxy = vx .* vy
    vyy = vy .* vy
    v2 = cat(vxx, vxy, vxy, vyy; dims = 3)
    v2 = reshape(v2, n, n, 2, 2)

    ## Quadractic terms in spectral space
    q = fft(v2, (1, 2))
    qx, qy = eachslice(q; dims = 4)

    ## Compute partial derivatives in spectral space
    ∂x = 2.0f0π * im * params.k
    ∂y = 2.0f0π * im * reshape(params.k, 1, :)
    q = @. -∂x * qx - ∂y * qy

    ## Zero out high wave-numbers (is this necessary?)
    q = [q[1:(params.Kf), 1:(params.Kf), :] z(params.Kf, 2Kz, 2) q[1:(params.Kf), (params.Kf + 2Kz + 1):end, :]
         z(2Kz, params.Kf, 2) z(2Kz, 2Kz, 2) z(2Kz, params.Kf, 2)
         q[(params.Kf + 2Kz + 1):end, 1:(params.Kf), :] z(params.Kf, 2Kz, 2) q[(params.Kf + 2Kz + 1):end, (params.Kf + 2Kz + 1):end, :]]

    q
end

# `F` computes the unprojected momentum right hand side $\hat{F}$. It also
# includes the closure term (if any).
function F_NS(u, params)
    q = Q(u, params)
    du = @. q - params.prefactor_F * u + params.f
    du
end

# The projector $P$ uses pre-assembled matrices.
function project(u, params)
    ux, uy = eachslice(u; dims = 3)
    dux = @. params.Pxx * ux + params.Pxy * uy
    duy = @. params.Pxy * ux + params.Pyy * uy
    cat(dux, duy; dims = 3)
end

#########################################
#----------------------------------------
# For plotting, the spatial vorticity can be useful. It is given by
#
# $$
# \omega = -\frac{\partial u_x}{\partial y} + \frac{\partial u_y}{\partial x},
# $$
#
# which becomes
#
# $$
# \hat{\omega} = 2 \pi \mathrm{i} k \times u = - 2 \pi \mathrm{i} k_y u_x + 2 \pi \mathrm{i} k_x u_y
# $$
#
# in spectral space.

function vorticity(u, params)
    ∂x = 2.0f0π * im * params.k
    ∂y = 2.0f0π * im * reshape(params.k, 1, :)
    ux, uy = eachslice(u; dims = 3)
    ω = @. -∂y * ux + ∂x * uy
    real.(ifft(ω))
end
