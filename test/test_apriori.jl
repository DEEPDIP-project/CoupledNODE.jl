using Test
using Random: Random
using IncompressibleNavierStokes: IncompressibleNavierStokes as INS
using JLD2: load, @save
using CoupledNODE: cnn, create_loss_priori, mean_squared_error, loss_priori_lux, train
NS = Base.get_extension(CoupledNODE, :NavierStokes)
using Lux: Lux
using Optimization: Optimization
using OptimizationOptimisers: OptimizationOptimisers
using Adapt

T = Float32
rng = Random.Xoshiro(123)
ig = 1 # index of the LES grid to use.

# Load the data
data = load("test_data/data_train.jld2", "data_train")
params = load("test_data/params_data.jld2", "params")
test_data = load("test_data/data_test.jld2", "data_test")

d = D = params.D

@testset "A-priori (CPU)" begin
    # Build LES setups and assemble operators
    setups = map(params.nles) do nles
        x = ntuple(α -> LinRange(T(0.0), T(1.0), nles + 1), params.D)
        INS.Setup(; x = x, Re = params.Re)
    end

    # Create io_arrays
    io_priori = NS.create_io_arrays_priori(data, setups[ig])
    test_io_priori = NS.create_io_arrays_priori(test_data, setups[ig])

    # Dataloader priori
    dataloader_prior = NS.create_dataloader_prior(io_priori; batchsize = 10, rng)
    train_data_priori = dataloader_prior()
    u, c = dataloader_prior()
    @test size(u) == (16, 16, 2, 10)
    @test size(c) == (16, 16, 2, 10)

    # Creation of the model: NN closure
    closure, θ,
    st = cnn(;
        T = T,
        D = D,
        data_ch = D,
        radii = [3, 3],
        channels = [2, 2],
        activations = [tanh, identity],
        use_bias = [false, false],
        rng
    )

    # Give the CNN a test run
    test_output = Lux.apply(closure, u, θ, st)[1]
    @test !isnothing(test_output) # Check that the output is not nothing

    # Loss in the Lux format
    loss_value = loss_priori_lux(closure, θ, st, train_data_priori)
    @test isfinite(loss_value[1]) # Check that the loss value is finite

    # Define the callback
    callbackstate_val,
    callback_val = NS.create_callback(
        closure, θ, test_io_priori, loss_priori_lux, st, batch_size = 30,
        rng = rng, do_plot = true, plot_train = false)

    # Training (via Lux)
    loss,
    tstate = train(closure, θ, st, dataloader_prior, loss_priori_lux;
        nepochs = 15, ad_type = Optimization.AutoZygote(),
        alg = OptimizationOptimisers.Adam(0.1), cpu = true, callback = nothing)

    # Check that the training loss is finite
    @test isfinite(loss)

    # The trained parameters at the end of the training are:
    θ_priori = tstate.parameters
    @test !isnothing(θ_priori) # Check that the trained parameters are not nothing
end

@testset "A-priori (GPU)" begin
    if !CUDA.functional()
        @testset "CUDA not available" begin
            @test true
        end
        return
    end

    # Use gpu device
    backend = CUDABackend()
    CUDA.allowscalar(false)
    device = x -> adapt(CuArray, x)

    # Build LES setups and assemble operators
    setups = map(params.nles) do nles
        x = ntuple(α -> LinRange(T(0.0), T(1.0), nles + 1), params.D)
        INS.Setup(; x = x, Re = params.Re, backend = backend)
    end

    # Create io_arrays
    io_priori = NS.create_io_arrays_priori(data, setups[ig], device)
    test_io_priori = NS.create_io_arrays_priori(test_data, setups[ig], device)

    # Dataloader priori
    dataloader_prior = NS.create_dataloader_prior(
        io_priori; batchsize = 10, rng = rng, device = device)
    train_data_priori = dataloader_prior()
    u, c = dataloader_prior()
    @test is_on_gpu(u)
    @test is_on_gpu(c)

    # Creation of the model: NN closure
    closure, θ,
    st = cnn(;
        T = T,
        D = D,
        data_ch = D,
        radii = [3, 3],
        channels = [2, 2],
        activations = [tanh, identity],
        use_bias = [false, false],
        rng = rng,
        use_cuda = true
    )
    @test is_on_gpu(θ.layer_4.weight) # Check that the parameters are on the GPU

    # Give the CNN a test run
    test_output = Lux.apply(closure, u, θ, st)[1]
    @test !isnothing(test_output) # Check that the output is not nothing
    @test is_on_gpu(test_output) # Check that the output is on the GPU

    # Loss in the Lux format
    loss_value = loss_priori_lux(closure, θ, st, train_data_priori)
    @test isfinite(loss_value[1]) # Check that the loss value is finite

    # Define the callback
    callbackstate_val,
    callback_val = NS.create_callback(
        closure, θ, test_io_priori, loss_priori_lux, st, batch_size = 30,
        rng = rng, do_plot = true, plot_train = false, device = device)

    # Training (via Lux)
    loss,
    tstate = train(closure, θ, st, dataloader_prior, loss_priori_lux;
        nepochs = 15, ad_type = Optimization.AutoZygote(),
        alg = OptimizationOptimisers.Adam(0.1), cpu = false, callback = nothing)

    # Check that the training loss is finite
    @test isfinite(loss)

    # The trained parameters at the end of the training are:
    θ_priori = tstate.parameters
    @test !isnothing(θ_priori) # Check that the trained parameters are not nothing
    @test is_on_gpu(θ_priori.layer_4.weight) # Check that the trained parameters are on the GPU
end
