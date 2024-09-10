# For an example of pure NeuralClosure+SciML, visit IncompressibleNavierStokes.jl/lib/NeuralClosure/test/examplerun.jl
# OK Luisa: here you have the workflow of NeuralCLosure with INS. Now the challenge is going to be to adapt this to the SciML framework.
# I would like to keep the data generation using Syver's code, but the training, loss and error calculation we may have an oportunity to define in CoupledNODE.
# Of course remember we are limiting ourselves to only using Zygote.
using CairoMakie
import Random
import IncompressibleNavierStokes as INS
import NeuralClosure as NC

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

"""
Create ``(\\bar{u}, c)`` pairs for training.
The io_arrays will also have the initial condition, just as the data.
Diference with NeuralClosure.create_io_arrays is that this one has a place-holder for boundary conditions.
"""
# not sure if we need this one. I think we can use the one from NeuralClosure. -> To discuss.
function create_io_arrays(data, setups)
    nsample = length(data)
    ngrid, nfilter = size(data[1].data)
    nt = length(data[1].t)
    T = eltype(data[1].t)
    map(CartesianIndices((ngrid, nfilter))) do I
        ig, ifil = I.I
        (; dimension, N, Iu) = setups[ig].grid
        D = dimension()
        u = zeros(T, N..., D, nt, nsample)
        c = zeros(T, N..., D, nt, nsample)
        for is in 1:nsample, it in 1:nt, α in 1:D
            copyto!(
                view(u, :, :, α, it, is),
                data[is].data[ig, ifil].u[it][α]
            )
            copyto!(
                view(c, :, :, α, it, is),
                data[is].data[ig, ifil].c[it][α]
            )
        end
        (; u = reshape(u, N..., D, :), c = reshape(c, N..., D, :)) # squeeze time and sample dimensions
    end
end

# Create input/output arrays for a-priori training (ubar vs c)
io = create_io_arrays(data, setups); # modified version that has also place-holder for boundary conditions
io_nc = NC.create_io_arrays(data, setups) # original syver version, without extra padding
ig = 1 #index of the LES grid to use.

# check that the data is correctly reshaped
a = data[1].data[ig].u[2][1]
# data[nsample.data[igrid,ifilter],u[t][dim as ux:1, uy:2]
b = io[ig].u[:, :, 1, 2]
# io[igrid].u[nx, ny, dim as ux:1, time*nsample]
a == b
c = io_nc[ig].u[:, :, 1, 2]
import CoupledNODE: NavierStokes as CN_NS
CN_NS.NN_to_INS(io_nc[ig].u[:, :, :, 2:2], setups[ig])[1]
CN_NS.INS_to_NN(data[1].data[ig].u[2], setups[ig])[:, :, 1, 1]

u0_NN = io_nc[ig].u[:, :, :, 2:2]
u0_INS = CN_NS.NN_to_INS(u0_NN, setups[ig])

# * Creation of the model: NN closure
import CoupledNODE: cnn
closure, θ, st = cnn(;
    setup = setups[ig],
    radii = [3, 3],
    channels = [2, 2],
    activations = [tanh, identity],
    use_bias = [false, false],
    rng
)

# Give the CNN a test run
import Lux
Lux.apply(closure, io_nc[ig].u[:, :, :, 1:1], θ, st)[1]
model_debug = Lux.Experimental.@debug_mode closure
Lux.apply(model_debug, io_nc[ig].u[:, :, :, 1:10], θ, st)

# * Right-hand-side out-of-place : not needed in a-priori fitting
#dudt_nn = CN_NS.create_right_hand_side_with_closure(setups[ig], INS.psolver_spectral(setups[ig]), closure, st)
#dudt_nn(u0_NN, θ, T(0)) # correct, what is designed for 
#dudt_nn(stack(data[ig].data[1].u[1]), nothing , T(0)) # fails, not design for this data structure (old way)

# * dataloader priori
dataloader = NC.create_dataloader_prior(io_nc[ig]; batchsize = 10, rng)
train_data = dataloader()
size(train_data[1]) # bar{u} filtered
size(train_data[2]) # c commutator error

# * loss a priori
import CoupledNODE: create_loss_priori, mean_squared_error
loss_priori = create_loss_priori(mean_squared_error, closure)
# this created function can be called: loss_priori((x, y), θ, st) where x: input to model (\bar{u}), y: label (c), θ: params of NN, st: state of NN.
loss_priori(closure, θ, st, train_data) # check that the loss is working

# this is not correct
mseloss = Lux.GenericLossFunction((ŷ, y) -> abs2(ŷ - y))
mseloss(closure, θ, st, train_data)
another_wrong_loss = Lux.MSELoss()
another_wrong_loss(closure, θ, st, train_data)
another_wrong_loss(train_data...)

# let's define a loss that calculates correctly and in the Lux format
function loss_lux_style(model, ps, st, (x, y))
    ŷ, st_ = model(x, ps, st)
    loss = sum(abs2, ŷ .- y) / sum(abs2, y)
    return loss, st_, (; y_pred = ŷ)
end
loss_lux_style(closure, θ, st, train_data)

## old way of training
import CoupledNODE: callback
import Optimization, OptimizationOptimisers
optf = Optimization.OptimizationFunction(
    (u, p) -> loss_priori(closure, u, st, train_data), # u here is the optimization variable (θ params of NN)
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

## not yet working
import CoupledNODE: train
import Optimization, OptimizationOptimisers
println("training")
tstate = Lux.Training.TrainState(closure, θ, st, OptimizationOptimisers.Adam(0.1))
_, loss, stats, tstate = Lux.Training.single_train_step!(
    Optimization.AutoZygote(), loss_lux_style, train_data, tstate)

train(closure, θ, st, dataloader, loss_lux_style;
    nepochs = 100, ad_type = Optimization.AutoZygote(),
    alg = OptimizationOptimisers.Adam(0.1), cpu = true)

# * A posteriori dataloader
# indeed the ioarrays are not useful here, what a bummer! We should come up with a format that would be useful for both a-priori and a-posteriori training. 
# here we do not use io_arrays, those were nice for a priori because had \bar{u}, c.

# in this function we cr4eate io_arrays with the following differences:
# - we do not need the commutator error c.
# - we need the time and we keep the initial condition
# - we do not have the boundary conditions extra elements
# - put time dimension in the end, since SciML also does
function create_io_arrays_posteriori(data, setups)
    nsample = length(data)
    ngrid, nfilter = size(data[1].data)
    nt = length(data[1].t) - 1
    T = eltype(data[1].t)
    map(CartesianIndices((ngrid, nfilter))) do I
        ig, ifil = I.I
        (; dimension, N, Iu) = setups[ig].grid
        D = dimension()
        u = zeros(T, (N .- 2)..., D, nsample, nt + 1)
        t = zeros(T, nsample, nt + 1)
        ifield = ntuple(Returns(:), D)
        for is in 1:nsample, it in 1:(nt + 1), α in 1:D
            copyto!(
                view(u, ifield..., α, is, it),
                view(data[is].data[ig, ifil].u[it][α], Iu[α])
            )
            copyto!(
                view(t, is, :),
                data[is].t
            )
        end
        (; u = u, t = t)
    end
end
io_post = create_io_arrays_posteriori(data, setups)
(n, _, dim, samples, nsteps) = size(io_post[ig].u) # (nles, nles, D, samples, tsteps+1)
(samples, nsteps) = size(io_post[ig].t)
#select a random sample
io_post[ig].u[:, :, :, rand(1:samples), :]
io_post[ig].t[2, :]

# Let's explore Syver's way of handling the data for a-posteriori fitting
trajectories = [(; u = d.data[ig].u, d.t) for d in data]
size(trajectories[3].u) # 101 timesteps each sample.
size(rand(snaps_out))
nunroll = 5
it = 1:nunroll
snaps = (; u = data[1].data[ig].u[it], t = data[1].t[it])
size(snaps.u) # nunroll timesteps
dataloader_post = NC.create_dataloader_post(trajectories; nunroll = nunroll, rng)
dataloader_post().u
dataloader_post().t

# Luisa: can we create a dataloader that has the data in a nicer (NN) format?
function create_dataloader_post_Luisa(io_array; nunroll = 10, device = identity, rng)
    function dataloader()
        (n, _, dim, samples, nt) = size(io_array.u) # expects that the io_array will be for a i_grid
        @assert nt ≥ nunroll
        istart = rand(rng, 1:(nt - nunroll))
        it = istart:(istart + nunroll)
        isample = rand(rng, 1:samples)
        (; u = view(io_array.u, :, :, :, isample, it), t = io_array.t[isample, it])
    end
end
dataloader_luisa = create_dataloader_post_Luisa(io_post[ig]; nunroll = nunroll, rng)
dataloader_luisa().u
dataloader_luisa().t

# option 1: data from Syver, rhs how simone used to define it
function rhs(setup, psolver, closure, st)
    function right_hand_side(u, p, t)
        u = eachslice(u; dims = ndims(u))
        u = (u...,)
        u = INS.apply_bc_u(u, t, setup)
        F = INS.momentum(u, nothing, t, setup)
        F = F .+ CN_NS.NN_to_INS(
            Lux.apply(closure, CN_NS.INS_to_NN(u, setup), p, st)[1][:, :, :, 1:1], setup)
        F = INS.apply_bc_u(F, t, setup; dudt = true)
        PF = INS.project(F, setup; psolver)
        PF = INS.apply_bc_u(PF, t, setup; dudt = true)
        cat(PF[1], PF[2]; dims = 3)
    end
end
dudt_nn = rhs(setups[ig], INS.psolver_spectral(setups[ig]), closure, st)
# how the rhs could be called:
dudt_nn(stack(dataloader_post().u[1]), θ, dataloader_post().t[1])

# option 2: Luisa's dataloader, io_arrays and rhs
dudt_nn2 = CN_NS.create_right_hand_side_with_closure(
    setups[ig], INS.psolver_spectral(setups[ig]), closure, st)
example2 = dataloader_luisa()
dudt_nn2(example2.u[:, :, :, 1], θ, example2.t[1]) # trick of compatibility: keep always last dimension (time*sample)

# Define the loss (a-posteriori)
import Zygote
import DifferentialEquations: ODEProblem, solve, Tsit5
# option 1
example = dataloader_post()
x0 = example.u[1]
stack(x0)
y = example.u
tspan = [example.t[1, 1], example.t[end, 1]]
dt = example.t[2, 1] - example.t[1, 1]
saveat_loss = [i * dt for i in 1:length(y)]
tspan
prob = ODEProblem(dudt_nn, stack(x0), tspan, θ)
pred = Array(solve(prob, Tsit5(); u0 = stack(x0), p = θ, dt = dt, adaptive = false))
pred
stack(y[2:end])
stack(y[2:end]) - pred # not possible
# We would need to handle the data differently in the loss function to be able to compare the results.

# option 2
tspan = [example2.t[1], example2.t[end]]
dt = example2.t[2] - example2.t[1]
prob2 = ODEProblem(dudt_nn2, example2.u[:, :, :, 1], tspan, θ)
pred = Array(solve(
    prob2, Tsit5(); u0 = example2.u[:, :, :, 1:1], p = θ, dt = dt, adaptive = false))
pred[:, :, :, 1, 2:end] - example2.u[:, :, :, 2:end]
pred[:, :, :, 1, 1] == example2.u[:, :, :, 1] # Sci-ML also keeps the initial condition.

function loss_posteriori(model, p, st, data)
    u, t = data
    x = u[:, :, :, 1:1]
    y = u[:, :, :, 2:end] # remember to discard sol at the initial time step
    #dt = params.Δt
    dt = t[2] - t[1]
    #saveat_loss = [i * dt for i in 1:length(y)]
    tspan = [t[1], t[end]]
    prob = ODEProblem(dudt_nn2, x, tspan, p)
    pred = Array(solve(prob, Tsit5(); u0 = x, p = p, dt = dt, adaptive = false))
    # remember that the first element of pred is the initial condition (SciML)
    return T(sum(abs2, y - pred[:, :, :, 1, 2:end]) / sum(abs2, y))
end

# let's define a loss that calculates correctly and in the Lux format
function loss_posteriori_lux_style(model, ps, st, (x, y))
    ŷ, st_ = model(x, ps, st)
    loss = sum(abs2, ŷ - y) / sum(abs2, y)
    return loss, st_, (; y_pred = ŷ)
end

# train a-posteriori: single data point
train_data_posteriori = dataloader_luisa()
optf = Optimization.OptimizationFunction(
    (u, p) -> loss_posteriori(closure, u, st, train_data_posteriori), # u here is the optimization variable (θ params of NN)
    Optimization.AutoZygote()
)
optprob = Optimization.OptimizationProblem(optf, θ)
result_posteriori = Optimization.solve(
    optprob,
    OptimizationOptimisers.Adam(0.1);
    callback = callback,
    maxiters = 50,
    progress = true
)
θ_posteriori = result_posteriori.u

loss_posteriori_lux_style(closure, θ_posteriori, st, train_data_posteriori)

# try with Lux
tstate = Lux.Training.TrainState(
    closure, θ_posteriori, st, OptimizationOptimisers.Adam(0.1))
_, loss, stats, tstate = Lux.Training.single_train_step!(
    Optimization.AutoZygote(), loss_lux_style, train_data_posteriori, tstate)

Lux.Training.compute_gradients(
    Optimization.AutoZygote(), loss_posteriori, train_data_posteriori, tstate)

# A-posteriori training Syver
θ_posteriori = let
    setup = setups[ig]
    psolver = INS.psolver_spectral(setup)
    loss = NC.create_loss_post(; setup, psolver, model.closure, nupdate = 1)
    snaps = [(; u = d.data[ig].u, d.t) for d in data]
    d = NC.create_dataloader_post(snaps; nunroll = 5, rng)
    θ = copy(model.θ₀)
    opt = Optimisers.Adam(1.0e-3)
    optstate = Optimisers.setup(opt, θ)
    it = 1:5
    snaps = (; u = data[1].data[ig].u[it], t = data[1].t[it])
    (; callbackstate, callback) = NC.create_callback(
        NC.create_relerr_post(;
            data = snaps,
            setup,
            psolver,
            closure_model = NC.wrappedclosure(model.closure, setup),
            nupdate = 2
        );
        θ,
        displayref = false,
        display_each_iteration = true # Set to `true` if using CairoMakie
    )
    (; optstate, θ, callbackstate) = NC.train(
        [d], loss, optstate, θ; niter = 3, ncallback = 1, callbackstate, callback)
    #keep the best set of parameters
end
