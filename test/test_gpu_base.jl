
using Test
using Lux, LuxCUDA, ComponentArrays, CUDA, Random

@testset "GPU base CNN" begin
    gpu = Lux.gpu_device()

    function cnn(;
            T = Float32,
            D,
            data_ch,
            radii,
            channels,
            activations,
            use_bias,
            rng = Random.default_rng()
    )
        r, c, σ, b = radii, channels, activations, use_bias

        dev = Lux.gpu_device()

        @warn "*******Using $(dev) "

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
            collocate,
            padder,
            # convolutional layers
            (Conv(
                 ntuple(α -> 2r[i] + 1, D),
                 c[i] => c[i + 1],
                 σ[i];
                 use_bias = b[i],
                 init_weight = glorot_uniform_T
             ) for i in eachindex(r)
            )...,
            decollocate
        )
        chain = Lux.Chain(layers...)
        params, state = Lux.setup(rng, chain)
        state = state |> dev
        params = ComponentArray(params) |> dev
        (chain, params, state)
    end

    function interpolate(A, D, dir)
        (i, a) = A
        if i > D
            return a  # Nothing to interpolate for extra layers
        end
        staggered = a .+ circshift(a, ntuple(x -> x == i ? dir : 0, D))
        staggered ./ 2
    end

    function collocate(u)
        D = ndims(u) - 2
        slices = eachslice(u; dims = D + 1)
        staggered_slices = map(x -> interpolate(x, D, 1), enumerate(slices))
        stack(staggered_slices; dims = D + 1)
    end

    """
    Interpolate closure force from volume centers to volume faces.
    """
    function decollocate(u)
        D = ndims(u) - 2
        slices = eachslice(u; dims = D + 1)
        staggered_slices = map(x -> interpolate(x, D, -1), enumerate(slices))
        stack(staggered_slices; dims = D + 1)
    end

    # Parameters for the CNN
    T = Float32
    rng = Random.Xoshiro(123)
    D = 2

    # Initialize the CNN model
    chain, params,
    state = cnn(;
        T = T,
        D = D,
        data_ch = D,
        radii = [3, 3],
        channels = [2, 2],
        activations = [tanh, identity],
        use_bias = [false, false],
        rng = rng
    )

    # Example input data
    input_data = rand(Float32, 32, 32, 2, 1) |> gpu

    # Apply the chain
    output = chain(input_data, params, state)

    # Test
    @test !isnothing(output)
    @test eltype(output[1]) == Float32
    @test typeof(input_data) == typeof(output[1])
end
