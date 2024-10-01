using Lux: Lux
using Random: Random
using IncompressibleNavierStokes: IncompressibleNavierStokes as INS
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
        setup,
        radii,
        channels,
        activations,
        use_bias,
        rng = Random.default_rng()
)
    r, c, σ, b = radii, channels, activations, use_bias
    (; T, grid) = setup
    (; dimension) = grid
    D = dimension()

    # Weight initializer
    glorot_uniform_T(rng::Random.AbstractRNG, dims...) = Lux.glorot_uniform(rng, T, dims...)

    # Make sure there are two force fields in output
    @assert c[end] == D

    # Add input channel size
    c = [D; c]

    # Create convolutional closure model
    layers = (
    # Some convolutional layers
        (Lux.Conv(
             ntuple(α -> 2r[i] + 1, D),
             c[i] => c[i + 1],
             σ[i];
             use_bias = b[i],
             init_weight = glorot_uniform_T,
             pad = (ntuple(α -> 2r[i] + 1, D) .- 1) .÷ 2
         ) for i in eachindex(r)
    )...,
    )
    chain = Lux.Chain(layers...)
    params, state = Lux.setup(rng, chain)
    (chain, ComponentArray(params), state)
end
