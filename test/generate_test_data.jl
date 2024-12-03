using IncompressibleNavierStokes: IncompressibleNavierStokes as INS
using JLD2: jldsave
using Random: Random

T = Float32
rng = Random.Xoshiro(123)

# Generate the data using NeuralClosure
# Add NeuralClosure from the git repo, until it is added to the public repo
using Pkg
Pkg.add(url = "https://github.com/DEEPDIP-project/NeuralClosure.jl.git")
using NeuralClosure: NeuralClosure as NC

# Number of simulations to generate for each grid
Nsim_train = 3
Nsim_test = 2

# Parameters
params = (;
    D = 2,
    Re = T(1e3),
    lims = (T(0.0), T(1.0)),
    nles = [16, 32],
    ndns = 64,
    filters = (NC.FaceAverage(),),
    tburn = T(5e-2),
    tsim = T(0.5),
    savefreq = 1,
    Δt = T(5e-3), create_psolver = INS.psolver_spectral,
    icfunc = (setup, psolver, rng) -> INS.random_field(
        setup, zero(eltype(setup.grid.x[1])); kp = 20, psolver, rng),
    rng
)

data_train = [NC.create_les_data(; params...) for _ in 1:Nsim_train];
data_test = [NC.create_les_data(; params...) for _ in 1:Nsim_test];

@test data_train isa Array
@test data_test isa Array
@test params isa NamedTuple

#save data
jldsave("test_data/data_train.jld2"; data_train)
jldsave("test_data/data_test.jld2"; data_test)
jldsave("test_data/params_data.jld2"; params)

@test isfile("test_data/data_train.jld2")
@test isfile("test_data/data_test.jld2")
@test isfile("test_data/params_data.jld2")
