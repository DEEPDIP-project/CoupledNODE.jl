# For an example of pure NeuralClosure+SciML, visit IncompressibleNavierStokes.jl/lib/NeuralClosure/test/examplerun.jl
# Goal: adapt the workflow of NeuralCLosure with INS to the SciML framework.
# Remember we are limiting ourselves to only using Zygote.
using CairoMakie
using Random: Random
using IncompressibleNavierStokes: IncompressibleNavierStokes as INS
using NeuralClosure: NeuralClosure as NC

# Parameters
T = Float32
ArrayType = Array
rng = Random.Xoshiro(123)
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

# Generate the data
data = [NC.create_les_data(; params...) for _ in 1:3]

# Build LES setups and assemble operators
setups = map(params.nles) do nles
    x = ntuple(α -> LinRange(T(0.0), T(1.0), nles[α] + 1), params.D)
    INS.Setup(x...; params.Re)
end

# create io_arrays
ig = 1 #index of the LES grid to use.
io_nc = NC.create_io_arrays(data, setups) # original version from syver

# * Creation of the model: NN closure
using CoupledNODE: cnn
closure, θ, st = cnn(;
    setup = setups[ig],
    radii = [3, 3],
    channels = [2, 2],
    activations = [tanh, identity],
    use_bias = [false, false],
    rng
)

# Give the CNN a test run
using Lux: Lux
Lux.apply(closure, io_nc[ig].u[:, :, :, 1:1], θ, st)[1]

# * dataloader priori
dataloader_prior = NC.create_dataloader_prior(io_nc[ig]; batchsize = 10, rng)
train_data_priori = dataloader_prior()
size(train_data_priori[1]) # bar{u} filtered
size(train_data_priori[2]) # c commutator error

# * loss a priori
using CoupledNODE: create_loss_priori, mean_squared_error
loss_priori = create_loss_priori(mean_squared_error, closure)
# this created function can be called: loss_priori((x, y), θ, st) where x: input to model (\bar{u}), y: label (c), θ: params of NN, st: state of NN.
loss_priori(closure, θ, st, train_data_priori) # check that the loss is working

# * loss in the Lux format
using CoupledNODE: loss_priori_lux
loss_priori_lux(closure, θ, st, train_data_priori)

# * old way of training
using CoupledNODE: callback
using Optimization: Optimization
using OptimizationOptimisers: OptimizationOptimisers
optf = Optimization.OptimizationFunction(
    (u, p) -> loss_priori(closure, u, st, train_data_priori), # u here is the optimization variable (θ params of NN)
    Optimization.AutoZygote())
optprob = Optimization.OptimizationProblem(optf, θ)
result_priori = Optimization.solve(
    optprob,
    OptimizationOptimisers.Adam(0.1);
    callback = callback,
    maxiters = 50,
    progress = true
)
θ_priori = result_priori.u
# with this approach, we have the problem that we cannot loop trough the data. 

# another option of callback
using CoupledNODE: create_stateful_callback
callbackstate, callback_2 = create_stateful_callback(θ)

# * new way of training (via Lux)
using CoupledNODE: train
loss, tstate = train(closure, θ, st, dataloader_prior, loss_priori_lux;
    nepochs = 50, ad_type = Optimization.AutoZygote(),
    alg = OptimizationOptimisers.Adam(0.1), cpu = true, callback = callback_2)
# the trained parameters at the end of the training are: 
θ_priori = tstate.parameters
# and the best parameters found during training are:
θ_priori_best = callbackstate.θmin

# * A posteriori io_arrays 
# Here we cannot use io_arrays, those were nice for a priori because had \bar{u}, c.
# And time and sample dimension were squeezed. Here we want to do trajectory fitting thus we need the time sequence.

# in this function we create io_arrays with the following carachteristics:
# - we do not need the commutator error c.
# - we need the time and we keep the initial condition
# - we do not have the boundary conditions extra elements
# - put time dimension in the end, since SciML also does
using CoupledNODE.NavierStokes: create_io_arrays_posteriori
io_post = create_io_arrays_posteriori(data, setups)

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

# * Define the right hand side of the ODE
using CoupledNODE.NavierStokes: create_right_hand_side_with_closure
dudt_nn2 = create_right_hand_side_with_closure(
    setups[ig], INS.psolver_spectral(setups[ig]), closure, st)

# * Define the loss (a-posteriori)
using Zygote: Zygote
using DifferentialEquations: ODEProblem, solve, Tsit5
# for testing purposes: debugging the rhs and ODEProblem interaction
example2 = dataloader_posteriori()
example2.u
dudt_nn2(example2.u[:, :, :, 1], θ, example2.t[1]) # no tricks needed!
tspan = [example2.t[1], example2.t[end]]
dt = example2.t[2] - example2.t[1]
prob2 = ODEProblem(dudt_nn2, example2.u[:, :, :, 1], tspan, θ)
pred = Array(solve(
    prob2, Tsit5(); u0 = example2.u[:, :, :, 1], p = θ, dt = dt, adaptive = false))
pred[:, :, :, 2:end] - example2.u[:, :, :, 2:end]
pred[:, :, :, 1] == example2.u[:, :, :, 1] # Sci-ML also keeps the initial condition.
# end - for testing purposes

# * Define the loss (a-posteriori) - old way
function loss_posteriori(model, p, st, data)
    u, t = data
    griddims = Zygote.@ignore ((:) for _ in 1:(ndims(u) - 2))
    x = u[griddims..., :, 1]
    y = u[griddims..., :, 2:end] # remember to discard sol at the initial time step
    #dt = params.Δt
    dt = t[2] - t[1]
    #saveat_loss = [i * dt for i in 1:length(y)]
    tspan = [t[1], t[end]]
    prob = ODEProblem(dudt_nn2, x, tspan, p)
    pred = Array(solve(prob, Tsit5(); u0 = x, p = p, dt = dt, adaptive = false))
    # remember that the first element of pred is the initial condition (SciML)
    return T(sum(
        abs2, y[griddims..., :, 1:(size(pred, 4) - 1)] - pred[griddims..., :, 2:end]) /
             sum(abs2, y))
end

# * train a-posteriori: old way single data point
train_data_posteriori = dataloader_posteriori()
optf = Optimization.OptimizationFunction(
    (u, p) -> loss_posteriori(closure, u, st, train_data_posteriori), # u here is the optimization variable (θ params of NN)
    Optimization.AutoZygote()
)
optprob = Optimization.OptimizationProblem(optf, θ)
result_posteriori = Optimization.solve(
    optprob,
    OptimizationOptimisers.Adam(0.1);
    callback = callback,
    maxiters = 10,
    progress = true
)
θ_posteriori = result_posteriori.u

# * package loss
using CoupledNODE: create_loss_post_lux
loss_posteriori_lux = create_loss_post_lux(dudt_nn2; sciml_solver = Tsit5())
loss_posteriori_lux(closure, θ, st, train_data_posteriori)

# * training via Lux
loss, tstate = train(closure, θ, st, dataloader_posteriori, loss_posteriori_lux;
    nepochs = 10, ad_type = Optimization.AutoZygote(),
    alg = OptimizationOptimisers.Adam(0.1), cpu = true, callback = callback)

# the trained params are:
θ_posteriori = tstate.parameters
