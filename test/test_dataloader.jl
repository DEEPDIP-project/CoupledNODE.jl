using Test
using Adapt
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

T = Float32
rng = Random.Xoshiro(123)
ig = 1 # index of the LES grid to use.
nunroll = 5

# Load the data
data = load("test_data/data_train.jld2", "data_train")
params = load("test_data/params_data.jld2", "params")
test_data = load("test_data/data_test.jld2", "data_test")
test_data_INS = load("test_data/data_test_INS.jld2", "data_test")

# Build LES setups and assemble operators
setups = map(params.nles) do nles
    x = ntuple(α -> LinRange(T(0.0), T(1.0), nles + 1), params.D)
    INS.Setup(; x = x, Re = params.Re)
end

@testset "A-priori vs INS" begin
    test_io = NS.create_io_arrays_priori(test_data, setups[ig])
    test_io_INS = NS.INS_create_io_arrays_priori(test_data_INS, setups)[1]
    @test size(test_io_INS.u)[1:3] == size(test_io.u)[1:3]
    @test size(test_io_INS.c)[4] == size(test_io.c)[4]+2
end

@testset "A-posteriori vs INS" begin
    test_io = NS.create_io_arrays_posteriori(test_data, setups[ig])
    test_io_INS = NS.INS_create_io_arrays_posteriori(test_data_INS, setups)[1]
    @test size(test_io_INS.u)[1:4] == size(test_io.u)[1:4]
    @test size(test_io_INS.u)[5] == size(test_io.u)[5]+1
end

@testset "A-posteriori (CPU)" begin
    # A posteriori io_arrays
    io_post = NS.create_io_arrays_posteriori(data, setups[ig])

    # Create dataloader containing trajectories with the specified nunroll
    dataloader_posteriori = NS.create_dataloader_posteriori(
        io_post; nunroll = nunroll, nsamples = 1, rng)
    u, t = dataloader_posteriori()
    @test size(u) == (18, 18, 2, 1, 6)
    @test size(t) == (1, 6)

    dataloader_posteriori = NS.create_dataloader_posteriori(
        io_post; nunroll = nunroll, nsamples = 3, rng)
    u, t = dataloader_posteriori()
    @test size(u) == (18, 18, 2, 3, 6)
    @test size(t) == (3, 6)
end

@testset "A-posteriori (GPU)" begin
    if !CUDA.functional()
        @testset "No GPU available" begin
            return
        end
    end
    device = x -> adapt(CuArray{T}, x)

    # A posteriori io_arrays
    io_post = NS.create_io_arrays_posteriori(data, setups[ig])

    # Create dataloader containing trajectories with the specified nunroll
    dataloader_posteriori = NS.create_dataloader_posteriori(
        io_post; nunroll = nunroll, nsamples = 1, device = device, rng)
    u, t = dataloader_posteriori()
    @test size(u) == (18, 18, 2, 1, 6)
    @test size(t) == (1, 6)
    @test isa(u, CuArray) # Check that the training data is on the GPU
    @test isa(t, CuArray) # Check that the training data is on the GPU

    dataloader_posteriori = NS.create_dataloader_posteriori(
        io_post; nunroll = nunroll, nsamples = 3, device = device, rng)
    u, t = dataloader_posteriori()
    @test size(u) == (18, 18, 2, 3, 6)
    @test size(t) == (3, 6)
    @test isa(u, CuArray) # Check that the training data is on the GPU
    @test isa(t, CuArray) # Check that the training data is on the GPU
end
