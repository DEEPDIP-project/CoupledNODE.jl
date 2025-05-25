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
using SciMLSensitivity

T = Float32
rng = Random.Xoshiro(123)
ig = 1 # index of the LES grid to use.
nunroll = 5

# Load the data
data = load("test_data/data_train.jld2", "data_train")
params = load("test_data/params_data.jld2", "params")
test_data = load("test_data/data_test.jld2", "data_test")

d = D = params.D
griddims = ((:) for _ in 1:D)
inside = ((:) for _ in 1:D)

sensealgs = [nothing, QuadratureAdjoint(), GaussAdjoint(),
    BacksolveAdjoint(), InterpolatingAdjoint()]
timings = Vector{Tuple{Any, Float64}}() # Store (sensealg, lux_t)
for SENSEALG_i in sensealgs
    @testset "(CPU) loss with $SENSEALG_i" begin

        # Build LES setups and assemble operators
        setups = map(params.nles) do nles
            x = ntuple(α -> LinRange(T(0.0), T(1.0), nles + 1), params.D)
            INS.Setup(; x = x, Re = params.Re)
        end

        # A posteriori io_arrays
        io_post = NS.create_io_arrays_posteriori(data, setups[ig])
        test_io_post = NS.create_io_arrays_posteriori(test_data, setups[ig])

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
        # Define the right hand side of the ODE
        dudt_nn2 = NS.create_right_hand_side_with_closure(
            setups[ig], INS.psolver_spectral(setups[ig]), closure, st)

        # Create dataloader containing trajectories with the specified nunroll
        dataloader_posteriori = NS.create_dataloader_posteriori(
            io_post; nunroll = nunroll, nsamples = 2, rng)
        u, t = dataloader_posteriori()
        train_data_posteriori = dataloader_posteriori()

        # Define the loss (a-posteriori)
        loss_posteriori_lux = create_loss_post_lux(
            dudt_nn2,
            griddims,
            inside;
            sensealg = SENSEALG_i
        )
        loss_value = loss_posteriori_lux(closure, θ, st, train_data_posteriori)
        @test isfinite(loss_value[1]) # Check that the loss value is finite

        # Callback function
        θ_posteriori = θ

        # Training via Lux
        _, _,
        _,
        _ = @timed train(
            closure, θ_posteriori, st, dataloader_posteriori, loss_posteriori_lux;
            nepochs = 1, ad_type = Optimization.AutoZygote(),
            alg = OptimizationOptimisers.Adam(0.01), cpu = true, callback = nothing)

        lux_result, lux_t,
        lux_mem,
        _ = @timed train(
            closure, θ_posteriori, st, dataloader_posteriori, loss_posteriori_lux;
            nepochs = 50, ad_type = Optimization.AutoZygote(),
            alg = OptimizationOptimisers.Adam(0.01), cpu = true, callback = nothing)

        push!(timings, (SENSEALG_i, lux_t))

        loss, tstate = lux_result
        # Check that the training loss is finite
        @test isfinite(loss)

        # The trained parameters at the end of the training are:
        θ_posteriori = tstate.parameters
        @test !isnothing(θ_posteriori) # Check that the trained parameters are not nothing
    end
end

# After the loop, sort and print the timings
sorted_timings = sort(timings; by = x -> x[2])
@warn "Ranking of (CPU) sensealgs by training time (fastest to slowest):"
for (alg, t) in sorted_timings
    @info "SenseAlg: $(alg), Time: $(t) seconds"
end

timings = Vector{Tuple{Any, Float64}}() # Reset timings for GPU tests
sensealgs = [nothing, GaussAdjoint(), BacksolveAdjoint(), InterpolatingAdjoint()]
for SENSEALG_i in sensealgs
    @testset "A-posteriori (GPU)" begin
        if !CUDA.functional()
            @testset "CUDA not available" begin
                @test true
            end
            return
        end

        # Helper function to check if a variable is on the GPU
        function is_on_gpu(x)
            return x isa CuArray
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

        # Define the right hand side of the ODE
        dudt_nn2 = NS.create_right_hand_side_with_closure(
            setups[ig], INS.psolver_spectral(setups[ig]), closure, st)

        # Define the loss (a-posteriori)
        train_data_posteriori = dataloader_posteriori()
        loss_posteriori_lux = create_loss_post_lux(
            dudt_nn2,
            griddims,
            inside;
            sensealg = SENSEALG_i
        )
        loss_value = loss_posteriori_lux(closure, θ, st, train_data_posteriori)
        @test isfinite(loss_value[1]) # Check that the loss value is finite

        θ_posteriori = θ

        # Training via Lux
        _, _,
        _,
        _ = @timed train(
            closure, θ_posteriori, st, dataloader_posteriori, loss_posteriori_lux;
            nepochs = 1, ad_type = Optimization.AutoZygote(),
            alg = OptimizationOptimisers.Adam(0.01), cpu = false, callback = nothing)
        lux_result, lux_t,
        lux_mem,
        _ = @timed train(
            closure, θ_posteriori, st, dataloader_posteriori, loss_posteriori_lux;
            nepochs = 50, ad_type = Optimization.AutoZygote(),
            alg = OptimizationOptimisers.Adam(0.01), cpu = false, callback = nothing)

        push!(timings, (SENSEALG_i, lux_t))

        loss, tstate = lux_result
        # Check that the training loss is finite
        @test isfinite(loss)

        # The trained parameters at the end of the training are:
        θ_posteriori = tstate.parameters
        @test !isnothing(θ_posteriori) # Check that the trained parameters are not nothing
        @test is_on_gpu(θ_posteriori.layer_4.weight) # Check that the trained parameters are on the GPU
    end
end

sorted_timings = sort(timings; by = x -> x[2])
@warn "Ranking of (GPU) sensealgs by training time (fastest to slowest):" begin
    for (alg, t) in sorted_timings
        @info "SenseAlg: $(alg), Time: $(t) seconds"
    end
end
