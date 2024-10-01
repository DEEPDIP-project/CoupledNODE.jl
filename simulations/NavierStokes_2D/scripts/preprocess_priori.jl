using Random: Random
using IncompressibleNavierStokes: IncompressibleNavierStokes as INS

T = Float32
ArrayType = Array
rng = Random.Xoshiro(123)

# Load the data
using JLD2: load
data = load("simulations/NavierStokes_2D/data/data.jld2", "data")
params = load("simulations/NavierStokes_2D/data/params_data.jld2", "params")

# Build LES setups and assemble operators
setups = map(params.nles) do nles
    x = ntuple(α -> LinRange(T(0.0), T(1.0), nles[α] + 1), params.D)
    INS.Setup(x...; params.Re)
end

# create io_arrays
using CoupledNODE.NavierStokes: create_io_arrays_priori
io_priori = create_io_arrays_priori(data, setups) # original version from syver

# * dataloader priori
using CoupledNODE.NavierStokes: create_dataloader_prior
dataloader_prior = create_dataloader_prior(io_priori[ig]; batchsize = 10, rng)
train_data_priori = dataloader_prior()
size(train_data_priori[1]) # bar{u} filtered
size(train_data_priori[2]) # c commutator error
