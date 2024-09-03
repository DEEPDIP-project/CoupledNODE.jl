import Lux, Random
import IncompressibleNavierStokes as INS
import NeuralClosure as NC
import ComponentArrays: ComponentArray
import NNlib: pad_circular, pad_repeat

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
        NC.collocate,  # Put inputs in pressure points
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
        NC.decollocate # Differentiate output to velocity points
    )
    chain = Lux.Chain(layers...)
    params, state = Lux.setup(rng, chain)
    (chain, ComponentArray(params), state)
end

"""
Interpolate velocity components to volume centers.
"""
function collocate(u)
    sz..., D = size(u)
    if D == 2
        a = selectdim(u, 3, 1)
        b = selectdim(u, 3, 2)
        a = (a .+ circshift(a, (1, 0))) ./ 2
        b = (b .+ circshift(b, (0, 1))) ./ 2
        a = reshape(a, sz..., 1)
        b = reshape(b, sz..., 1)
        cat(a, b; dims = 3)
    elseif D == 3
        a = selectdim(u, 4, 1)
        b = selectdim(u, 4, 2)
        c = selectdim(u, 4, 3)
        a = (a .+ circshift(a, (1, 0, 0))) ./ 2
        b = (b .+ circshift(b, (0, 1, 0))) ./ 2
        c = (c .+ circshift(c, (0, 0, 1))) ./ 2
        a = reshape(a, sz..., 1)
        b = reshape(b, sz..., 1)
        c = reshape(c, sz..., 1)
        cat(a, b, c; dims = 4)
    end
end

"""
Interpolate closure force from volume centers to volume faces.
"""
function decollocate(u)
    sz..., D = size(u)
    if D == 2
        a = selectdim(u, 3, 1)
        b = selectdim(u, 3, 2)
        a = (a .+ circshift(a, (-1, 0))) ./ 2
        b = (b .+ circshift(b, (0, -1))) ./ 2
        a = reshape(a, sz..., 1)
        b = reshape(b, sz..., 1)
        cat(a, b; dims = 3)
    elseif D == 3
        a = selectdim(u, 4, 1)
        b = selectdim(u, 4, 2)
        c = selectdim(u, 4, 3)
        a = (a .+ circshift(a, (-1, 0, 0))) ./ 2
        b = (b .+ circshift(b, (0, -1, 0))) ./ 2
        c = (c .+ circshift(c, (0, 0, -1))) ./ 2
        a = reshape(a, sz..., 1)
        b = reshape(b, sz..., 1)
        c = reshape(c, sz..., 1)
        cat(a, b, c; dims = 4)
    end
end
