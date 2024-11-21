using Random: Random
using IncompressibleNavierStokes: IncompressibleNavierStokes as INS

T = Float32
rng = Random.Xoshiro(123)

# Load the data
using JLD2: load
#data = load("simulations/NavierStokes_2D/data/data_train_PaperDC_float64.jld2", "data_train")
#params = load("simulations/NavierStokes_2D/data/params_data_PaperDC_float64.jld2", "params")

data = load("/var/scratch/lorozco/INS_old/lib/PaperDC/output/postanalysis/data_train.jld2", "data_train")
#params = load("../data/params_data_PaperDC_float64.jld2", "params")

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

# create io_arrays
using CoupledNODE.NavierStokes: create_io_arrays_priori
io_priori = create_io_arrays_priori(data, setups) # original version from syver

# * dataloader priori
using CoupledNODE.NavierStokes: create_dataloader_prior
dataloader_prior = create_dataloader_prior(io_priori[ig]; batchsize = 100, rng)
train_data_priori = dataloader_prior()
size(train_data_priori[1]); # bar{u} filtered
size(train_data_priori[2]); # c commutator error

# Load the test data
#test_data = load("simulations/NavierStokes_2D/data/data_test_PaperDC_float32.jld2", "data_test")
test_data = load("/var/scratch/lorozco/INS_old/lib/PaperDC/output/postanalysis/data_valid.jld2", "data_valid")

test_io_post = create_io_arrays_priori(test_data, setups)
