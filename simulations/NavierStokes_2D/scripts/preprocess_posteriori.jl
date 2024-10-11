using Random: Random
using IncompressibleNavierStokes: IncompressibleNavierStokes as INS

T = Float32
ArrayType = Array
rng = Random.Xoshiro(123)

# Load the data
using JLD2: load
data = load("simulations/NavierStokes_2D/data/data_train.jld2", "data_train")
params = load("simulations/NavierStokes_2D/data/params_data.jld2", "params")

# Build LES setups and assemble operators
setups = map(params.nles) do nles
    x = ntuple(α -> LinRange(T(0.0), T(1.0), nles[α] + 1), params.D)
    INS.Setup(x...; params.Re)
end

# * A posteriori io_arrays 
using CoupledNODE.NavierStokes: create_io_arrays_posteriori
io_post = create_io_arrays_posteriori(data, setups)

# Example of dimensions and how to operate with io_arrays_posteriori
ig = 1
(n, _, dim, samples, nsteps) = size(io_post[ig].u) # (nles, nles, D, samples, tsteps+1)
(samples, nsteps) = size(io_post[ig].t)
# Example: how to select a random sample
io_post[ig].u[:, :, :, rand(1:samples), :]
io_post[ig].t[2, :]

# * Create dataloader containing trajectories with the specified nunroll
using CoupledNODE.NavierStokes: create_dataloader_posteriori
nunroll = 5
dataloader_posteriori = create_dataloader_posteriori(io_post; nunroll = nunroll, rng)

# Load the test data
test_data = load("simulations/NavierStokes_2D/data/data_test.jld2", "data_test")
test_io_post = create_io_arrays_posteriori(test_data, setups)
