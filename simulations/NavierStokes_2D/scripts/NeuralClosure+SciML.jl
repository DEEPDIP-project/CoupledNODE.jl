# For an example of pure NeuralClosure+SciML, visit IncompressibleNavierStokes.jl/lib/NeuralClosure/test/examplerun.jl
# Goal: adap the workflow of NeuralCLosure with INS to the SciML framework.
# Remember we are limiting ourselves to only using Zygote.
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

#include("../../../src/equations/NavierStokes_utils.jl") # for development
import CoupledNODE: IO_padded_to_IO_nopad, NN_padded_to_INS, NN_padded_to_NN_nopad

# Create input/output arrays for a-priori training (ubar vs c)
io = create_io_arrays(data, setups); # modified version that has also place-holder for boundary conditions
ig = 1 #index of the LES grid to use.
io_nc = IO_padded_to_IO_nopad(io, setups) # equivalent to original syver version, without extra padding

# check that the data is correctly reshaped
a = data[1].data[ig].u[2][1]
# data[nsample].data[igrid,ifilter].u[t][dim as ux:1, uy:2]
b = io[ig].u[:, :, 1, 2]
# io[igrid].u[nx, ny, dim as ux:1, time*nsample]
a == b
c = io_nc[ig].u[:, :, 1, 2]

NN_padded_to_INS(io[ig].u[:, :, :, 2:2], setups[ig])
NN_padded_to_NN_nopad(io[ig].u[:, :, :, 2:2], setups[ig])

zu0_NN = io[ig].u[:, :, :, 2:2]

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

# * dataloader priori
dataloader_prior = NC.create_dataloader_prior(io_nc[ig]; batchsize = 10, rng)
train_data_priori = dataloader_prior()
size(train_data_priori[1]) # bar{u} filtered
size(train_data_priori[2]) # c commutator error

# * loss a priori (similar to Syver's)
import CoupledNODE: create_loss_priori, mean_squared_error
loss_priori = create_loss_priori(mean_squared_error, closure)
# this created function can be called: loss_priori((x, y), θ, st) where x: input to model (\bar{u}), y: label (c), θ: params of NN, st: state of NN.
loss_priori(closure, θ, st, train_data_priori) # check that the loss is working

# let's define a loss that calculates correctly and in the Lux format
import CoupledNODE: loss_priori_lux
loss_priori_lux(closure, θ, st, train_data_priori)

## old way of training
import CoupledNODE: callback
import Optimization, OptimizationOptimisers
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

# atempt to create a nice callback
function create_callback(
        θ,
        err_function = nothing,
        callbackstate = (; θmin = θ, θmin_e = θ, loss_min = eltype(θ)(Inf),
            emin = eltype(θ)(Inf), hist = Point2f[]),
        displayref = false,
        display_each_iteration = true,
        filename = nothing
)
    istart = isempty(callbackstate.hist) ? 0 : Int(callbackstate.hist[end][1])
    obs = Observable([Point2f(0, 0)])
    fig = lines(obs; axis = (; title = "Error", xlabel = "step"))
    displayref && hlines!([1.0f0]; linestyle = :dash)
    obs[] = callbackstate.hist
    display(fig)
    function callback(θ, loss)
        if err_function !== nothing
            e = err_function(θ)
            #@info "Iteration $i \terror: $e"
            e < state.emin && (state = (; callbackstate..., θmin_e = θ, emin = e))
        end
        hist = push!(copy(callbackstate.hist), Point2f(length(callbackstate.hist), loss))
        obs[] = hist
        autolimits!(fig.axis)
        display_each_iteration && display(fig)
        isnothing(filename) || save(filename, fig)
        callbackstate = (; callbackstate..., hist)
        loss < callbackstate.loss_min &&
            (callbackstate = (; callbackstate..., θmin = θ, loss_min = loss))
        callbackstate
    end
    (; callbackstate, callback)
end
callbackstate, callback_2 = create_callback(θ)

# new way of training (via Lux)
#include("../../../src/train.jl")
import CoupledNODE: train
import Optimization, OptimizationOptimisers
loss, tstate = train(closure, θ, st, dataloader_prior, loss_priori_lux;
    nepochs = 50, ad_type = Optimization.AutoZygote(),
    alg = OptimizationOptimisers.Adam(0.1), cpu = true, callback = callback_2)
# the trained parameters at the end of the training are: 
θ_priori = tstate.parameters
# and the best parameters found during training are:
θ_priori_best = callbackstate.θmin

# * A posteriori dataloader 
# Here we cannot use io_arrays, those were nice for a priori because had \bar{u}, c.
# And time and sample dimension were squeezed. Here we want to do trajectory fitting thus we need the time sequence.

# in this function we create io_arrays with the following carachteristics:
# - we do not need the commutator error c.
# - we need the time and we keep the initial condition
# - we do not have the boundary conditions extra elements
# - put time dimension in the end, since SciML also does
import CoupledNODE: create_io_arrays_posteriori
io_post = create_io_arrays_posteriori(data, setups)
(n, _, dim, samples, nsteps) = size(io_post[ig].u) # (nles, nles, D, samples, tsteps+1)
(samples, nsteps) = size(io_post[ig].t)
#select a random sample
io_post[ig].u[:, :, :, rand(1:samples), :]
io_post[ig].t[2, :]

# Let's explore Syver's way of handling the data for a-posteriori fitting
trajectories = [(; u = d.data[ig].u, d.t) for d in data]
size(trajectories[3].u) # 101 timesteps each sample.
nunroll = 5
it = 1:nunroll
snaps = (; u = data[1].data[ig].u[it], t = data[1].t[it])
size(snaps.u) # nunroll timesteps
dataloader_post = NC.create_dataloader_post(trajectories; nunroll = nunroll, rng)
dataloader_post().u
dataloader_post().t

# Luisa: can we create a dataloader that has the data in a nicer (NN) format?
import CoupledNODE: create_dataloader_posteriori
dataloader_posteriori = create_dataloader_posteriori(io_post[ig]; nunroll = nunroll, rng)
dataloader_posteriori().u
dataloader_posteriori().t

# option 2: Luisa's dataloader, io_arrays and rhs
import CoupledNODE: create_right_hand_side_with_closure_minimal_copy
dudt_nn2 = create_right_hand_side_with_closure_minimal_copy(
    setups[ig], INS.psolver_spectral(setups[ig]), closure, st)

# for testing purposes
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

# option 1: 
import CoupledNODE: create_right_hand_side_with_closure
#dudt_nn = create_right_hand_side_with_closure(
#    setups[ig], INS.psolver_spectral(setups[ig]), closure, st)
#dudt_nn(example2.u[:, :, :, 1], θ, example2.t[1])
import CoupledNODE: INS_to_NN
import NNlib: pad_circular
function dudt_nn1(u, p, t)
    # not sure if we should keep t as a parameter. t is only necessary for the INS functions when having Dirichlet BCs (time dependent)
    D = 2 # for the moment setting the dimension fixed to 2
    u = NN_padded_to_INS(u, setups[1])
    u = INS.apply_bc_u(u, t, setups[1])
    F = INS.momentum(u, nothing, t, setups[1])
    F_nopad = INS_to_NN(F, setups[1])
    F_nopad = F_nopad .+ Lux.apply(closure, INS_to_NN(u, setups[1]), p, st)[1][:, :, :, 1:1]
    F = NN_padded_to_INS(pad_circular(F_nopad, 1, dims = collect(1:D)), setups[1])
    F = INS.apply_bc_u(F, t, setups[1]; dudt = true)
    PF = INS.project(F, setups[1]; psolver = INS.psolver_spectral(setups[ig]))
    PF = INS.apply_bc_u(PF, t, setups[1]; dudt = true)
    INS_to_NN(PF, setups[1])
end
dudt_nn1(example2.u[:, :, :, 1], θ, example2.t[1])

# Define the loss (a-posteriori)
import Zygote
import DifferentialEquations: ODEProblem, solve, Tsit5

function loss_posteriori(model, p, st, data)
    u, t = data
    x = u[:, :, :, 1]
    y = u[:, :, :, 2:end] # remember to discard sol at the initial time step
    #dt = params.Δt
    dt = t[2] - t[1]
    #saveat_loss = [i * dt for i in 1:length(y)]
    tspan = [t[1], t[end]]
    prob = ODEProblem(dudt_nn2, x, tspan, p)
    pred = Array(solve(prob, Tsit5(); u0 = x, p = p, dt = dt, adaptive = false))
    # remember that the first element of pred is the initial condition (SciML)
    return T(sum(abs2, y[:, :, :, 1:(size(pred, 4) - 1)] - pred[:, :, :, 2:end]) /
             sum(abs2, y))
end

# train a-posteriori: single data point
using SciMLSensitivity
SciMLSensitivity.STACKTRACE_WITH_VJPWARN[] = true
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
    maxiters = 5,
    progress = true
)
θ_posteriori = result_posteriori.u

# test package loss
import CoupledNODE: create_loss_post_lux
loss_posteriori_lux = create_loss_post_lux(dudt_nn2; sciml_solver = Tsit5())
loss_posteriori_lux(closure, θ, st, train_data_posteriori)

# training via Lux
loss, tstate = train(closure, θ, st, dataloader_posteriori, loss_posteriori_lux;
    nepochs = 5, ad_type = Optimization.AutoZygote(),
    alg = OptimizationOptimisers.Adam(0.1), cpu = true, callback = callback)

# the trained params are now:
θ_posteriori = tstate.parameters
