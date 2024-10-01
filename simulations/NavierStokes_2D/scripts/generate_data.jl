using Random: Random
using IncompressibleNavierStokes: IncompressibleNavierStokes as INS

T = Float32
ArrayType = Array
rng = Random.Xoshiro(123)

# Generate the data using NeuralClosure
# locally in the environment of the project you can add this dependency as follows:
# julia
# ]
# dev "path_to_NeuralClosure"

using NeuralClosure: NeuralClosure as NC

# Parameters
params = (;
    D = 2,
    Re = T(1e3),
    tburn = T(5e-2),
    tsim = T(0.5),
    Δt = T(5e-3),
    nles = [(16, 16), (32, 32)],
    ndns = (64, 64),
    filters = (NC.FaceAverage(),),
    create_psolver = INS.psolver_spectral,
    icfunc = (setup, psolver, rng) -> INS.random_field(
        setup, zero(eltype(setup.grid.x[1])); kp = 20, psolver, rng),
    rng,
    savefreq = 1
)

data = [NC.create_les_data(; params...) for _ in 1:3]

# save data
using JLD2: jldsave
jldsave("simulations/NavierStokes_2D/data/data.jld2"; data)
jldsave("simulations/NavierStokes_2D/data/params_data.jld2"; params)
