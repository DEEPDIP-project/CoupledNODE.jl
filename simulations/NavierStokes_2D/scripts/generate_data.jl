using IncompressibleNavierStokes: IncompressibleNavierStokes as INS
using JLD2: jldsave
using Random: Random

T = Float64
rng = Random.Xoshiro(123)

using Pkg
Pkg.add(url = "https://github.com/DEEPDIP-project/NeuralClosure.jl.git")
using NeuralClosure: NeuralClosure as NC

# Number of simulations to generate for each grid
Nsim_train = 5
Nsim_test = 2

# Parameters
params = (;
    D = 2,
    Re = T(1e3),
    lims = (T(0.0), T(1.0)),
    nles = [32],
    ndns = 64,
    filters = (NC.FaceAverage(),),
    tburn = T(0.5),
    tsim = T(5),
    savefreq = 1,
    bodyforce = (dim, x, y, t) -> (dim == 1) * 5 * sinpi(8 * y),
    issteadybodyforce = true,
    Δt = T(5e-3), create_psolver = INS.psolver_spectral,
    icfunc = (setup,
        psolver,
        rng) -> INS.random_field(
        setup, zero(eltype(setup.grid.x[1])); kp = 20, psolver, rng),
    rng
)

data_train = [NC.create_les_data(; params...) for _ in 1:Nsim_train];
data_test = [NC.create_les_data(; params...) for _ in 1:Nsim_test];

#save data
jldsave("simulations/NavierStokes_2D/data/data_train.jld2"; data_train)
jldsave("simulations/NavierStokes_2D/data/data_test.jld2"; data_test)
jldsave("simulations/NavierStokes_2D/data/params_data.jld2"; params)
