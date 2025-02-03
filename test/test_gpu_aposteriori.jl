using Test
using Random: Random
using IncompressibleNavierStokes: IncompressibleNavierStokes as INS
using JLD2: load, @save
using CoupledNODE: cnn, train, create_loss_post_lux
NS = Base.get_extension(CoupledNODE, :NavierStokes)
using DifferentialEquations: ODEProblem, solve, Tsit5
using ComponentArrays: ComponentArray
using Lux: Lux
using LuxCUDA
using Adapt
using Optimization: Optimization
using OptimizationOptimisers: OptimizationOptimisers

# Define the test set
@testset "GPU A-posteriori" begin
    # Helper function to check if a variable is on the GPU
    function is_on_gpu(x)
        return x isa CuArray
    end
    T = Float32
    rng = Random.Xoshiro(123)
    ig = 1 # index of the LES grid to use.

    @test CUDA.functional() # Check that CUDA is available

    # Load the data
    data = load("test_data/data_train.jld2", "data_train")
    params = load("test_data/params_data.jld2", "params")

    # Use gpu device
    backend = CUDABackend()
    CUDA.allowscalar(false)
    device = x -> adapt(CuArray, x)

    # Build LES setups and assemble operators
    setups = map(params.nles) do nles
        x = ntuple(α -> LinRange(T(0.0), T(1.0), nles + 1), params.D)
        INS.Setup(; x = x, Re = params.Re)
    end

    # A posteriori io_arrays 
    io_post = NS.create_io_arrays_posteriori(data, setups, device)

    # Example of dimensions and how to operate with io_arrays_posteriori
    (n, _, dim, samples, nsteps) = size(io_post[ig].u) # (nles, nles, D, samples, tsteps+1)
    (samples, nsteps) = size(io_post[ig].t)
    # Example: how to select a random sample
    random_sample = io_post[ig].u[:, :, :, rand(1:samples), :]
    random_time = io_post[ig].t[2, :]

    # Create dataloader containing trajectories with the specified nunroll
    nunroll = 5
    dataloader_posteriori = NS.create_dataloader_posteriori(
        io_post[ig]; nunroll = nunroll, rng = rng, device = device)
    train_data_post = dataloader_posteriori()
    @test is_on_gpu(train_data_post[1]) # Check that the training data is on the GPU
    @test is_on_gpu(train_data_post[2]) # Check that the training data is on the GPU

    # Load the test data
    test_data = load("test_data/data_test.jld2", "data_test")
    test_io_post = NS.create_io_arrays_posteriori(test_data, setups)

    u = device(io_post[ig].u[:, :, :, 1, 1:10])
    #T = setups[1].T
    d = D = setups[1].grid.dimension()
    N = size(u, 1)

    # Define the CNN layers
    closure, θ, st = cnn(;
        T = T,
        D = D,
        data_ch = D,
        radii = [3, 3],
        channels = [2, 2],
        activations = [tanh, identity],
        use_bias = [false, false],
        rng
    )
    θ = device(θ)
    st = device(st)

    # Test and trigger the model
    test_output = Lux.apply(closure, u, θ, st)
    @test !isnothing(test_output) # Check that the output is not nothing
    @test is_on_gpu(u) # Check that the output is on the GPU
    @test is_on_gpu(test_output) # Check that the output is on the GPU

    # Define the right hand side of the ODE
    dudt_nn2 = NS.create_right_hand_side_with_closure(
        setups[ig], INS.psolver_spectral(setups[ig]), closure, st)

    # Define the loss (a-posteriori) 
    train_data_posteriori = dataloader_posteriori()
    loss_posteriori_lux = create_loss_post_lux(dudt_nn2; sciml_solver = Tsit5())
    loss_value = loss_posteriori_lux(closure, θ, st, train_data_posteriori)
    @test isfinite(loss_value[1]) # Check that the loss value is finite

    ## Callback function
    #callbackstate_val, callback_val = NS.create_callback(
    #    dudt_nn2, θ, test_io_post[ig], loss_posteriori_lux, st, nunroll = 3 * nunroll,
    #    rng = rng, do_plot = false, plot_train = false, device=device)
    #θ_posteriori = θ

    ## Training via Lux
    #lux_result, lux_t, lux_mem, _ = @timed train(
    #    closure, θ_posteriori, st, dataloader_posteriori, loss_posteriori_lux;
    #    nepochs = 50, ad_type = Optimization.AutoZygote(),
    #    alg = OptimizationOptimisers.Adam(0.01), cpu = false, callback = nothing)

    #loss, tstate = lux_result
    ## Check that the training loss is finite
    #@test isfinite(loss)

    ## The trained parameters at the end of the training are:
    #θ_posteriori = tstate.parameters
    #@test !isnothing(θ_posteriori) # Check that the trained parameters are not nothing
end
