using Random: Random
using IncompressibleNavierStokes: IncompressibleNavierStokes as INS

T = Float32
rng = Random.Xoshiro(123)

# Load the data
using JLD2: load
#data = load("simulations/NavierStokes_2D/data/data_train_PaperDC_float32.jld2", "data_train")
#params = load("simulations/NavierStokes_2D/data/params_data_PaperDC_float32.jld2", "params")

data = load("/var/scratch/lorozco/INS_old/lib/PaperDC/output/postanalysis/data_train.jld2", "data_train")
# Parameters
get_params(nlesscalar) = (;
    D = 2,
    Re = T(10_000),
    tburn = T(0.05),
    tsim = T(0.5),
    Δt = T(5e-5),
    nles = map(n -> (n, n), nlesscalar), # LES resolutions
    ndns = (n -> (n, n))(4096), # DNS resolution
    filters = (1, 2), 
    ArrayType,
    create_psolver = INS.psolver_spectral,
    icfunc = (setup, psolver, rng) ->
        random_field(setup, zero(eltype(setup.grid.x[1])); kp = 20, psolver, rng),
    rng,
)
params = (; get_params([64, 128, 256])..., tsim = T(0.5), savefreq = 10);

# Build LES setups and assemble operators
setups = map(params.nles) do nles
    x = ntuple(α -> LinRange(T(0.0), T(1.0), nles + 1), params.D)
    INS.Setup(; x = x, Re = params.Re)
end

# * A posteriori io_arrays 
using CoupledNODE.NavierStokes: create_io_arrays_posteriori
io_post = create_io_arrays_posteriori(data, setups)
ig = 1
# Example of dimensions and how to operate with io_arrays_posteriori
(n, _, dim, samples, nsteps) = size(io_post[ig].u) # (nles, nles, D, samples, tsteps+1)
(samples, nsteps) = size(io_post[ig].t)
# Example: how to select a random sample
io_post[ig].u[:, :, :, rand(1:samples), :]
io_post[ig].t[2, :]

# * Create dataloader containing trajectories with the specified nunroll
using CoupledNODE.NavierStokes: create_dataloader_posteriori
nunroll = 5
dataloader_posteriori = create_dataloader_posteriori(io_post[ig]; nunroll = nunroll, rng)

# Load the test data
#test_data = load("simulations/NavierStokes_2D/data/data_test_PaperDC_float32.jld2", "data_test")
test_data = load("/var/scratch/lorozco/INS_old/lib/PaperDC/output/postanalysis/data_valid.jld2", "data_valid")
test_io_post = create_io_arrays_posteriori(test_data, setups)
