using JLD2: jldsave
using Random: Random
using IncompressibleNavierStokes
using NeuralClosure
NS = Base.get_extension(CoupledNODE, :NavierStokes)

if CUDA.functional()
    backend = CUDABackend()
else
    backend = IncompressibleNavierStokes.CPU()
end

T = Float32
rng = Random.Xoshiro(123)

# Number of simulations to generate for each grid
Nsim_train = 3
Nsim_test = 2

# Parameters
params = (;
    D = 2,
    Re = T(1e3),
    lims = (T(0.0), T(1.0)),
    nles = [16],
    ndns = 64,
    filters = (FaceAverage(),),
    tburn = T(5e-2),
    tsim = T(0.2),
    savefreq = 1,
    Δt = T(5e-3), create_psolver = psolver_spectral,
    icfunc = (setup, psolver,
        rng) -> random_field(
        setup, zero(eltype(setup.grid.x[1])); kp = 20, psolver, rng),
    rng
)
@test params isa NamedTuple
jldsave("test_data/params_data.jld2"; params)
@test isfile("test_data/params_data.jld2")

if !isfile("test_data/data_train.jld2")
    data_train = [NS.create_les_data_projected(;
                      params...,
                      backend = backend
                  ) for _ in 1:Nsim_train];
    jldsave("test_data/data_train.jld2"; data_train)
    @test isfile("test_data/data_train.jld2")
end
if !isfile("test_data/data_test.jld2")
    data_test = [NS.create_les_data_projected(;
                     params...,
                     backend = backend
                 ) for _ in 1:Nsim_test];
    jldsave("test_data/data_test.jld2"; data_test)
    @test isfile("test_data/data_test.jld2")
end
if !isfile("test_data/data_test_INS.jld2")
    data_test = [create_les_data(; params...) for _ in 1:Nsim_test];
    @test data_test isa Array
    jldsave("test_data/data_test_INS.jld2"; data_test)
    @test isfile("test_data/data_test_INS.jld2")
end
