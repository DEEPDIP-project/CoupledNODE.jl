###########
# This file contains code that was removed from the main script for the sake of clarity
# It can be deleted once we are sure that we do not need it anymore

#**************************************************************************
# We can now generate the initial conditions for the Burgers equation. We will use a combination of sine waves to play the role of a smooth component, and a gaussian to induce the shock
using Interpolations, Statistics
function generate_initial_conditions(n_samples::Int, grids, gen_res = 5000)
    Nx = grids[1].nx
    x = range(0, stop = 2π, length = gen_res)
    xdns = range(0, stop = 2π, length = Nx + 1)
    xdns = xdns[1:(end - 1)]

    u0_list = Array{Float32, 2}(undef, Nx, n_samples)

    for j in 1:n_samples
        ## Smooth component (e.g., sine wave)
        #smooth_amplitude = 0.5
        #smooth_freq = 10.0 * rand() # replace 10 with the maximum multiple you want
        #smooth_phase_shift = rand()
        #u0 = smooth_amplitude * sin.(smooth_freq .* x .+ smooth_phase_shift)
        u0 = zeros(gen_res)

        # Gaussian shock component
        for s in 1:5
            shock_amplitude = rand() - 0.5
            shock_position = 2π * rand()
            shock_width = 1.6 * rand() + 0.4
            # impose periodicity
            for i in -5:5
                u0 .+= shock_amplitude *
                       exp.(-((x .- (shock_position + i * 2π)) .^ 2) / (2 * shock_width^2))
            end
        end

        # Shift such that it is centered around 0
        u0 .-= mean(u0)

        # Normalize the initial conditions
        u0 ./= maximum(abs.(u0))

        # and interpolate to the grid
        itp = LinearInterpolation(x, u0, extrapolation_bc = Periodic())

        u0_list[:, j] = itp[xdns]
    end

    return u0_list
end

# *************
# Bits for filtering
u0_les = zeros(Float32, nux_les, size(u0_dns, 2))
for i in 1:size(u0_dns, 2)
    itp = LinearInterpolation(xdns, u0_dns[:, i], extrapolation_bc = Periodic())
    u0_les[:, i] = itp[xles]
end
# -------
target_F = zeros(Float32, nux_les, size(all_u_dns)[2:end]...);
for i in 1:size(all_u_dns, 2)
    for t in 1:size(all_u_dns, 3)
        # Interepolate the DNS data to the LES grid
        itp = LinearInterpolation(xdns, all_u_dns[:, i, t], extrapolation_bc = Periodic())
        all_u_les[:, i, t] = itp[xles]
        # The target of the a-priori fitting is the filtered DNS force
        itp = LinearInterpolation(xdns, all_F_dns[:, i, t], extrapolation_bc = Periodic())
        target_F[:, i, t] = itp[xles]
    end
end
# -------
u_dns_test_filtered = zeros(Float32, nux_les, size(u_dns_test, 2), size(u_dns_test, 3));
for i in 1:size(u_dns_test, 2)
    for t in 1:size(u_dns_test, 3)
        itp = LinearInterpolation(xdns, u_dns_test[:, i, t], extrapolation_bc = Periodic())
        u_dns_test_filtered[:, i, t] = itp[xles]
    end
end
# -------
f_dns_filtered_test = zeros(Float32, size(f_les_test)...)
for i in 1:size(f_les_test, 2)
    for t in 1:size(f_les_test, 3)
        itp = LinearInterpolation(xdns, f_dns_test[:, i, t], extrapolation_bc = Periodic())
        f_dns_filtered_test[:, i, t] = itp[xles]
    end
end
# -------
u0_test_les = zeros(Float32, nux_les, size(u0_test, 2));
for i in 1:size(u0_test, 2)
    itp = LinearInterpolation(xdns, u0_test[:, i], extrapolation_bc = Periodic())
    u0_test_les[:, i] = itp[xles]
end

# *************
# Use a NN based on fully connected residual layers
# [!] this is very unlikely to work since it tends to predict discontinuous forces (very bad!)
#region
#using Lux
## define a residul block
#residual_block(n) = SkipConnection(
#    Chain(
#        Dense(n,n, init_weight=zeros32),
#        BatchNorm(n, leakyrelu),
#        Dropout(0.1),
#        Dense(n,n, init_weight=zeros32),
#        BatchNorm(n, leakyrelu),
#    ), +
#)
#NN_u = Chain(
#    u -> let u = u
#        # stack in the channel dimension
#        u = reshape(u, size(u, 1), size(u,2)*size(u, 3))
#        u
#    end,
#    residual_block(nux_les),
#    residual_block(nux_les),
#    Dense(nux_les, nux_les, init_weight=zeros32),
#    u -> let u = u
#        # reshape for meanpool
#        u = reshape(u, size(u,1), 1, size(u,2))
#    end,
#    AdaptiveMeanPool((Int(nux_les*0.35),)),
#    Upsample(:trilinear, size = (nux_les,)),
#    u -> let u = u
#        # stack in the channel dimension
#        u = reshape(u, size(u, 1), size(u,2)*size(u, 3))
#        u
#    end,
#)
#endregion

#*********************
# Compute the forces 
# 1 dns
f_dns_test = Array(f_dns(u_dns_test, θ_dns, st_dns)[1])
f_dns_test = reshape(f_dns_test, size(u_dns_test)...)
# 2 les over filtered solution
f_les_test = Array(f_les(u_dns_test_filtered, θ_les, st_les)[1])
f_les_test = reshape(f_les_test, size(u_les_test)...)
# 3 filtered f_dns
f_dns_filtered_test = Φ * reshape(
    f_dns_test, nux_dns, size(f_dns_test)[2] * size(f_dns_test)[3]);
f_dns_filtered_test = reshape(
    f_dns_filtered_test, nux_les, size(f_dns_test)[2], size(f_dns_test)[3]);
# 4 trained
f_trained_test = Array(f_CNODE(u_dns_test_filtered, θ, st)[1])
f_trained_test = reshape(f_trained_test, size(u_les_test)...)
# Here we chek if the trained les is better or not than the standard les
error_no_training = sum(abs.(f_dns_filtered_test - f_les_test))
error_training = sum(abs.(f_dns_filtered_test - f_trained_test))
# plot the forces
anim = Animation()
fig = plot(layout = (3, 1), size = (800, 300))
@gif for i in 1:2:size(u_trained_test, 3)
    p1 = plot(xdns, f_dns_test[:, 1, i], xlabel = "x", ylabel = "F",
        linetype = :steppre, label = "DNS")
    plot!(xles, f_dns_filtered_test[:, 1, i], linetype = :steppre, label = "Filtered DNS")
    plot!(xles, f_les_test[:, 1, i], linetype = :steppre, label = "LES")
    plot!(xles, f_trained_test[:, 1, i], linetype = :steppre, label = "Trained")
    p2 = plot(xdns, f_dns_test[:, 2, i], xlabel = "x", ylabel = "F",
        linetype = :steppre, legend = false)
    plot!(xles, f_dns_filtered_test[:, 2, i], linetype = :steppre, legend = false)
    plot!(xles, f_les_test[:, 2, i], linetype = :steppre, legend = false)
    plot!(xles, f_trained_test[:, 2, i], linetype = :steppre, legend = false)
    p3 = plot(xdns, f_dns_test[:, 3, i], xlabel = "x", ylabel = "F",
        linetype = :steppre, legend = false)
    plot!(xles, f_dns_filtered_test[:, 3, i], linetype = :steppre, legend = false)
    plot!(xles, f_les_test[:, 3, i], linetype = :steppre, legend = false)
    plot!(xles, f_trained_test[:, 3, i], linetype = :steppre, legend = false)
    title = "Time: $(round((i - 1) * saveat_shock, digits = 2))"
    fig = plot(p1, p2, p3, layout = (3, 1), title = title)
    frame(anim, fig)
end

# ****************
#region
#using NNlib, ComponentArrays, FFTW
#
#function create_model(chain, rng)
#    ## Create parameter vector and empty state
#    θ, state = Lux.setup(rng, chain)
#
#    ## Convert nested named tuples of arrays to a ComponentArray,
#    ## which behaves like a long vector
#    θ = ComponentArray(θ)
#
#    ## Convenience wrapper for empty state in input and output
#    m(v, θ) = first(chain(v, θ, state))
#
#    ## Return model and initial parameters
#    m, θ
#end
#struct FourierLayer{A,F} <: Lux.AbstractExplicitLayer
#    kmax::Int
#    cin::Int
#    cout::Int
#    σ::A
#    init_weight::F
#end
#
#FourierLayer(kmax, ch::Pair{Int,Int}; σ = identity, init_weight = glorot_uniform) =
#    FourierLayer(kmax, first(ch), last(ch), σ, init_weight)
#
#length(methods(Lux.initialparameters))
#
#Lux.initialparameters(rng::AbstractRNG, (; kmax, cin, cout, init_weight)::FourierLayer) = (;
#    spatial_weight = init_weight(rng, cout, cin),
#    spectral_weights = init_weight(rng, kmax + 1, cout, cin, 2),
#)
#Lux.initialstates(::AbstractRNG, ::FourierLayer) = (;)
#Lux.parameterlength((; kmax, cin, cout)::FourierLayer) =
#    cout * cin + (kmax + 1) * 2 * cout * cin
#Lux.statelength(::FourierLayer) = 0
#
### Pretty printing
#function Base.show(io::IO, (; kmax, cin, cout, σ)::FourierLayer)
#    print(io, "FourierLayer(", kmax)
#    print(io, ", ", cin, " => ", cout)
#    print(io, "; σ = ", σ)
#    print(io, ")")
#end
#
### One more method now
#length(methods(Lux.initialparameters))
#
### This makes FourierLayers callable
#function ((; kmax, cout, cin, σ)::FourierLayer)(x, params, state)
#    nx = size(x, 1)
#
#    ## Destructure params
#    ## The real and imaginary parts of R are stored in two separate channels
#    W = params.spatial_weight
#    W = reshape(W, 1, cout, cin)
#    R = params.spectral_weights
#    R = selectdim(R, 4, 1) .+ im .* selectdim(R, 4, 2)
#
#    ## Spatial part (applied point-wise)
#    y = reshape(x, nx, 1, cin, :)
#    y = sum(W .* y; dims = 3)
#    y = reshape(y, nx, cout, :)
#
#    ## Spectral part (applied mode-wise)
#    ##
#    ## Steps:
#    ##
#    ## - go to complex-valued spectral space
#    ## - chop off high wavenumbers
#    ## - multiply with weights mode-wise
#    ## - pad with zeros to restore original shape
#    ## - go back to real valued spatial representation
#    ikeep = 1:kmax+1
#    nkeep = kmax + 1
#    z = rfft(x, 1)
#    z = z[ikeep, :, :]
#    z = reshape(z, nkeep, 1, cin, :)
#    z = sum(R .* z; dims = 3)
#    z = reshape(z, nkeep, cout, :)
#    z = vcat(z, zeros(nx ÷ 2 + 1 - kmax - 1, size(z, 2), size(z, 3)))
#    z = irfft(z, nx, 1)
#
#    ## Outer layer: Activation over combined spatial and spectral parts
#    ## Note: Even though high wavenumbers are chopped off in `z` and may
#    ## possibly not be present in the input at all, `σ` creates new high
#    ## wavenumbers. High wavenumber functions may thus be represented using a
#    ## sequence of Fourier layers. In this case, the `y`s are the only place
#    ## where information contained in high input wavenumbers survive in a
#    ## Fourier layer.
#    w = σ.(z .+ y)
#
#    ## Fourier layer does not modify state
#    w, state
#end
#
#
#function create_fno(; channels, kmax, activations, rng, input_channels = (u -> u,))
#    ## Add number of input channels
#    channels = [length(input_channels); channels]
#
#    ## Model
#    create_model(
#        Chain(
#            ## Create singleton channel
#            #u -> reshape(u, size(u, 1), 1, size(u, 2)),
#
#            ## Create input channels
#            u -> hcat(map(i -> i(u), input_channels)...),
#
#            ## Some Fourier layers
#            (
#                FourierLayer(kmax[i], channels[i] => channels[i+1]; σ = activations[i]) for
#                i ∈ eachindex(kmax)
#            )...,
#
#            ## Put channels in first dimension
#            u -> permutedims(u, (2, 1, 3)),
#
#            ## Compress with a final dense layer
#            Dense(channels[end] => 2 * channels[end], gelu),
#            Dense(2 * channels[end] => 1; use_bias = false),
#
#            ## Put channels back after spatial dimension
#            u -> permutedims(u, (2, 1, 3)),
#
#            ## Remove singleton channel
#            u -> reshape(u, size(u, 1), size(u, 3)),
#        ),
#        rng,
#    )
#end
#
## ## Getting to business: Training and comparing closure models
##
## We now create a closure model. Note that the last activation is `identity`, as we
## don't want to restrict the output values. We can inspect the structure in the
## wrapped Lux `Chain`.
#
#
#m_fno, θ_fno = create_fno(;
#    channels = [5, 5, 5, 5],
#    kmax = [16, 16, 16, 8],
#    activations = [gelu, gelu, gelu, identity],
#    #input_channels = (u -> u, u -> u .^ 2),
#    input_channels = (u -> u,),
#    rng,
#)
#m_fno.chain
#
#NN_u = m_fno.chain
#endregion
