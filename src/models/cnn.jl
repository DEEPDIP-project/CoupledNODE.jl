using Lux
using NNlib: pad_circular
using Random: Random
using ComponentArrays: ComponentArray

"""
    cnn(; setup, radii, channels, activations, use_bias, channel_augmenter = identity, rng = Random.default_rng())

Constructs a convolutional neural network model `closure(u, θ)` that predicts the commutator error (i.e. closure).

# Arguments
- `setup`: IncompressibleNavierStokes.jl setup
- `radii`: An array (size n_layers) with the radii of the kernels for the convolutional layers. Kernels will be symmetrical of size `2r+1`.
- `channels`: An array (size n_layers) with channel sizes for the convolutional layers.
- `activations`: An array (size n_layers) with activation functions for the convolutional layers.
- `use_bias`: An array (size n_layers) with booleans indicating whether to use bias in each convolutional layer.
- `rng`: A random number generator (default: `Random.default_rng()`).

# Returns
A tuple `(chain, params, state)` where
- `chain`: The constructed Lux.Chain model.
- `params`: The parameters of the model.
- `state`: The state of the model.
"""
function cnn(;
        T = Float32,
        D,
        data_ch,
        radii,
        channels,
        activations,
        use_bias,
        rng = Random.default_rng(),
        use_cuda = false
)
    r, c, σ, b = radii, channels, activations, use_bias

    Cuda_ext = Base.get_extension(CoupledNODE, :CoupledNODECUDA)
    if !isnothing(Cuda_ext) && use_cuda
        interpolate_fn = Cuda_ext.gpu_interpolate
        dev = Cuda_ext.get_device()
    else
        interpolate_fn = cpu_interpolate
        dev = Lux.cpu_device()
    end

    # Weight initializer
    glorot_uniform_T(rng::Random.AbstractRNG, dims...) = glorot_uniform(rng, T, dims...)

    @assert length(c)==length(r)==length(σ)==length(b) "The number of channels, radii, activations, and use_bias must match"
    @assert c[end]==D "The number of output channels must match the data dimension"

    # Put the data channels at the beginning
    c = [data_ch; c]

    # Syver uses a padder layer instead of adding padding to the convolutional layers
    padder = ntuple(α -> (u -> pad_circular(u, sum(r); dims = α)), D)


    # Create convolutional closure model
    layers = (
        #u -> collocate(u, interpolate_fn),
        #padder,
        # convolutional layers
        (Conv(
             ntuple(α -> 2r[i] + 1, D),
             c[i] => c[i + 1],
             σ[i];
             use_bias = b[i],
             init_weight = glorot_uniform_T             #pad = (ntuple(α -> 2r[i] + 1, D) .- 1) .÷ 2
         ) for i in eachindex(r)
        )...,
        #u -> decollocate(u, interpolate_fn)
    )
    chain = Chain(layers...)
    params, state = Lux.setup(rng, chain) |> dev
    (chain, ComponentArray(params), state)
end

"""
Interpolate velocity components to volume centers.

TODO, D and dir can be parameters istead of arguments I think
"""
function cpu_interpolate(A, D, dir)
    @warn "Using CPU version of interpolate"
    (i, a) = A
    if i > D
        return a  # Nothing to interpolate for extra layers
    end
    staggered = a .+ circshift(a, ntuple(x -> x == i ? dir : 0, D))
    staggered ./ 2
end

function collocate(u, interpolate_fn=cpu_interpolate)
    D = ndims(u) - 2
    slices = eachslice(u; dims = D + 1)
    staggered_slices = map(x -> interpolate_fn(x, D, 1), enumerate(slices))
    stack(staggered_slices; dims = D + 1)
end

"""
Interpolate closure force from volume centers to volume faces.
"""
function decollocate(u, interpolate_fn=cpu_interpolate)
    D = ndims(u) - 2
    slices = eachslice(u; dims = D + 1)
    staggered_slices = map(x -> interpolate_fn(x, D, -1), enumerate(slices))
    stack(staggered_slices; dims = D + 1)
end
