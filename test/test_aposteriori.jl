using Test
using Random: Random
using IncompressibleNavierStokes: IncompressibleNavierStokes as INS
using JLD2: load, @save
using CoupledNODE: cnn, train, create_loss_post_lux
NS = Base.get_extension(CoupledNODE, :NavierStokes)
using DifferentialEquations: ODEProblem, solve, Tsit5
using ComponentArrays: ComponentArray
using Lux: Lux
using Optimization: Optimization
using OptimizationOptimisers: OptimizationOptimisers
using Adapt

T = Float32
rng = Random.Xoshiro(123)
ig = 1 # index of the LES grid to use.
nunroll = 5
dt = T(1e-4)

# Load the data
data = load("test_data/data_train.jld2", "data_train")
params = load("test_data/params_data.jld2", "params")
test_data = load("test_data/data_test.jld2", "data_test")

d = D = params.D
griddims = ((:) for _ in 1:D)
inside = ((:) for _ in 1:D)

@testset "A-posteriori (CPU)" begin

    # Build LES setups and assemble operators
    setups = map(params.nles) do nles
        x = ntuple(α -> LinRange(T(0.0), T(1.0), nles + 1), params.D)
        INS.Setup(; x = x, Re = params.Re)
    end

    # A posteriori io_arrays
    io_post = NS.create_io_arrays_posteriori(data, setups[ig])
    test_io_post = NS.create_io_arrays_posteriori(test_data, setups[ig])

    # Create dataloader containing trajectories with the specified nunroll
    dataloader_posteriori = NS.create_dataloader_posteriori(
        io_post; nunroll = nunroll, rng)
    u, t = dataloader_posteriori()

    # Define the CNN layers
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

    # Test and trigger the model
    test_output = closure(u[:, :, :, 1, :], θ, st)
    @test !isnothing(test_output) # Check that the output is not nothing

    # Define the right hand side of the ODE
    dudt_nn2 = NS.create_right_hand_side_with_closure(
        setups[ig], INS.psolver_spectral(setups[ig]), closure, st)

    # Define the loss (a-posteriori)
    train_data_posteriori = dataloader_posteriori()
    loss_posteriori_lux = create_loss_post_lux(
        dudt_nn2,
        griddims,
        inside,
        dt;
    )
    loss_value = loss_posteriori_lux(closure, θ, st, train_data_posteriori)
    @test isfinite(loss_value[1]) # Check that the loss value is finite

    # Callback function
    callbackstate_val,
    callback_val = NS.create_callback(
        dudt_nn2, θ, test_io_post, loss_posteriori_lux, st, nunroll = 3 * nunroll,
        rng = rng, do_plot = true, plot_train = false)
    θ_posteriori = θ

    # Training via Lux
    lux_result, lux_t,
    lux_mem,
    _ = @timed train(
        closure, θ_posteriori, st, dataloader_posteriori, loss_posteriori_lux;
        nepochs = 50, ad_type = Optimization.AutoZygote(),
        alg = OptimizationOptimisers.Adam(0.01), cpu = true, callback = nothing)

    loss, tstate = lux_result
    # Check that the training loss is finite
    @test isfinite(loss)

    # The trained parameters at the end of the training are:
    θ_posteriori = tstate.parameters
    @test !isnothing(θ_posteriori) # Check that the trained parameters are not nothing
end

@testset "A-posteriori (GPU)" begin
    if !CUDA.functional()
        @testset "CUDA not available" begin
            @test true
        end
        return
    end

    # Use gpu device
    backend = CUDABackend()
    CUDA.allowscalar(false)
    device = x -> adapt(CuArray{Float32}, x)

    # Build LES setups and assemble operators
    setups = map(params.nles) do nles
        x = ntuple(α -> LinRange(T(0.0), T(1.0), nles + 1), params.D)
        INS.Setup(; x = x, Re = params.Re, backend = backend)
    end

    # A posteriori io_arrays
    io_post = NS.create_io_arrays_posteriori(data, setups[ig], device)
    test_io_post = NS.create_io_arrays_posteriori(test_data, setups[ig], device)

    # Create dataloader containing trajectories with the specified nunroll
    dataloader_posteriori = NS.create_dataloader_posteriori(
        io_post; nunroll = nunroll, rng = rng, device = device)
    u, t = dataloader_posteriori()
    @test is_on_gpu(u)
    @test is_on_gpu(t)

    # Define the CNN layers
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

    # Test and trigger the model
    CUDA.@allowscalar u0 = u[:, :, :, 1, :]
    test_output = Lux.apply(closure, u0, θ, st)[1]
    @test !isnothing(test_output) # Check that the output is not nothing
    @test is_on_gpu(u) # Check that the output is on the GPU
    @test is_on_gpu(test_output) # Check that the output is on the GPU

    # Define the right hand side of the ODE
    dudt_nn2 = NS.create_right_hand_side_with_closure(
        setups[ig], INS.psolver_spectral(setups[ig]), closure, st)

    # Define the loss (a-posteriori)
    train_data_posteriori = dataloader_posteriori()
    loss_posteriori_lux = create_loss_post_lux(
        dudt_nn2,
        griddims,
        inside,
        dt;
    )
    loss_value = loss_posteriori_lux(closure, θ, st, train_data_posteriori)
    @test isfinite(loss_value[1]) # Check that the loss value is finite

    # Callback function
    callbackstate_val,
    callback_val = NS.create_callback(
        dudt_nn2, θ, test_io_post, loss_posteriori_lux, st, nunroll = 3 * nunroll,
        rng = rng, do_plot = false, plot_train = false, device = device)
    θ_posteriori = θ

    # Training via Lux
    lux_result, lux_t,
    lux_mem,
    _ = @timed train(
        closure, θ_posteriori, st, dataloader_posteriori, loss_posteriori_lux;
        nepochs = 50, ad_type = Optimization.AutoZygote(),
        alg = OptimizationOptimisers.Adam(0.01), cpu = false, callback = nothing)

    loss, tstate = lux_result
    # Check that the training loss is finite
    @test isfinite(loss)

    # The trained parameters at the end of the training are:
    θ_posteriori = tstate.parameters
    @test !isnothing(θ_posteriori) # Check that the trained parameters are not nothing
    @test is_on_gpu(θ_posteriori.layer_4.weight) # Check that the trained parameters are on the GPU
end
