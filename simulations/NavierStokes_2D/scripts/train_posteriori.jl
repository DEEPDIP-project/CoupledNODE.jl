using Random: Random
using IncompressibleNavierStokes: IncompressibleNavierStokes as INS

T = Float32
ArrayType = Array
rng = Random.Xoshiro(123)
ig = 1 # index of the LES grid to use.
include("preprocess_posteriori.jl")

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

# * Define the right hand side of the ODE
using CoupledNODE.NavierStokes: create_right_hand_side_with_closure
dudt_nn2 = create_right_hand_side_with_closure(
    setups[ig], INS.psolver_spectral(setups[ig]), closure, st)

# * Define the loss (a-posteriori) - old way
using DifferentialEquations: ODEProblem, solve, Tsit5
function loss_posteriori(model, p, st, data)
    u, t = data
    griddims = axes(u)[1:(ndims(u) - 2)]
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
using CoupledNODE: callback
using Optimization: Optimization
using OptimizationOptimisers: OptimizationOptimisers
train_data_posteriori = dataloader_posteriori()
optf = Optimization.OptimizationFunction(
    (x, _) -> loss_posteriori(closure, x, st, train_data_posteriori), # x here is the optimization variable (θ params of NN)
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
using CoupledNODE: train
loss, tstate = train(closure, θ, st, dataloader_posteriori, loss_posteriori_lux;
    nepochs = 10, ad_type = Optimization.AutoZygote(),
    alg = OptimizationOptimisers.Adam(0.1), cpu = true, callback = callback)

# the trained params are:
θ_posteriori = tstate.parameters
