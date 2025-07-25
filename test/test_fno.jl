using Test
using Random: Random
using IncompressibleNavierStokes: IncompressibleNavierStokes as INS
using JLD2: load, @save
using CoupledNODE: cnn, create_loss_priori, mean_squared_error, loss_priori_lux, train,
                   create_loss_post_lux
NS = Base.get_extension(CoupledNODE, :NavierStokes)
using NeuralOperators
nsfno = Base.get_extension(CoupledNODE, :fno)
using Lux: Lux
using Optimization: Optimization
using OptimizationOptimisers: OptimizationOptimisers
using ComponentArrays: ComponentArray
using Adapt

T = Float32
rng = Random.Xoshiro(123)
ig = 1 # index of the LES grid to use.
nunroll = 3

# Load the data
data = load("test_data/data_train.jld2", "data_train")
params = load("test_data/params_data.jld2", "params")
test_data = load("test_data/data_test.jld2", "data_test")

d = D = params.D
griddims = ((:) for _ in 1:D)
inside = ((:) for _ in 1:D)
dt = T(1e-3)

@testset "FNO (CPU)" begin
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
    @info typeof(u)

    closure, θ,
    st = nsfno.fno_closure(
        T = T,
        chs = (2, 64, 64, 64, 2),
        activation = Lux.gelu,
        modes = (4, 4),
        rng = rng
    )
    test_output = closure(u, θ, st)[1]
    @test !isnothing(test_output) # Check that the output is not nothing
    @test size(test_output) == (16, 16, 2, 10) # Check that the output has the correct size

    @testset "A-priori" begin
        # Loss in the Lux format
        loss_value = loss_priori_lux(closure, θ, st, train_data_priori)
        @test isfinite(loss_value[1]) # Check that the loss value is finite

        # Training (via Lux)
        loss,
        tstate = train(closure, θ, st, dataloader_prior, loss_priori_lux;
            nepochs = 5, ad_type = Optimization.AutoZygote(),
            alg = OptimizationOptimisers.Adam(0.1), cpu = true, callback = nothing)

        # Check that the training loss is finite
        @test isfinite(loss)

        # The trained parameters at the end of the training are:
        θ_priori = tstate.parameters
        @test !isnothing(θ_priori) # Check that the trained parameters are not nothing
    end

    @testset "A-posteriori" begin
        @warn "A-posteriori training is not available for FNO"
        return
        # A posteriori io_arrays
        io_post = NS.create_io_arrays_posteriori(data, setups[ig])
        test_io_post = NS.create_io_arrays_posteriori(test_data, setups[ig])

        # Create dataloader containing trajectories with the specified nunroll
        dataloader_posteriori = NS.create_dataloader_posteriori(
            io_post; nunroll = nunroll, rng)

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

        θ_posteriori = θ

        # Training via Lux
        lux_result, lux_t,
        lux_mem,
        _ = @timed train(
            closure, θ_posteriori, st, dataloader_posteriori, loss_posteriori_lux;
            nepochs = 5, ad_type = Optimization.AutoZygote(),
            alg = OptimizationOptimisers.Adam(0.01), cpu = true, callback = nothing)

        loss, tstate = lux_result
        # Check that the training loss is finite
        @test isfinite(loss)

        # The trained parameters at the end of the training are:
        θ_posteriori = tstate.parameters
        @test !isnothing(θ_posteriori) # Check that the trained parameters are not nothing
    end
end

@testset "FNO (GPU)" begin
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
    st = nsfno.fno_closure(
        chs = (2, 64, 64, 64, 2),
        activation = Lux.gelu,
        modes = (4, 4),
        use_cuda = true,
        rng = rng
    )
    @test isa(first(values(θ)).weight, CuArray)

    # Give the CNN a test run
    test_output = closure(u, θ, st)[1]
    @test !isnothing(test_output) # Check that the output is not nothing
    @test is_on_gpu(test_output) # Check that the output is on the GPU

    @testset "A-priori" begin
        # Loss in the Lux format
        loss_value = loss_priori_lux(closure, θ, st, train_data_priori)
        @test isfinite(loss_value[1]) # Check that the loss value is finite

        # Training (via Lux)
        loss,
        tstate = train(closure, θ, st, dataloader_prior, loss_priori_lux;
            nepochs = 5, ad_type = Optimization.AutoZygote(),
            alg = OptimizationOptimisers.Adam(0.1), cpu = false, callback = nothing)

        # Check that the training loss is finite
        @test isfinite(loss)

        # The trained parameters at the end of the training are:
        θ_priori = tstate.parameters
        @test !isnothing(θ_priori) # Check that the trained parameters are not nothing
        @test isa(first(values(θ_priori)).weight, CuArray)
    end

    @testset "A-posteriori" begin
        @warn "A-posteriori training is not available for FNO"
        return
        # A posteriori io_arrays
        io_post = NS.create_io_arrays_posteriori(data, setups[ig], device)
        test_io_post = NS.create_io_arrays_posteriori(test_data, setups[ig], device)

        # Create dataloader containing trajectories with the specified nunroll
        dataloader_posteriori = NS.create_dataloader_posteriori(
            io_post; nunroll = nunroll, rng, device = device)

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

        θ_posteriori = θ

        # Training via Lux
        lux_result, lux_t,
        lux_mem,
        _ = @timed train(
            closure, θ_posteriori, st, dataloader_posteriori, loss_posteriori_lux;
            nepochs = 5, ad_type = Optimization.AutoZygote(),
            alg = OptimizationOptimisers.Adam(0.01), cpu = true, callback = nothing)

        loss, tstate = lux_result
        # Check that the training loss is finite
        @test isfinite(loss)

        # The trained parameters at the end of the training are:
        θ_posteriori = tstate.parameters
        @test !isnothing(θ_posteriori) # Check that the trained parameters are not nothing
    end
end
