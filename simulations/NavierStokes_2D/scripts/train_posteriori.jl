using CoupledNODE: cnn, train, callback, create_loss_post_lux
using CoupledNODE.NavierStokes: create_right_hand_side_with_closure
using DifferentialEquations: ODEProblem, solve, Tsit5
using IncompressibleNavierStokes: IncompressibleNavierStokes as INS
using JLD2: @save
using Optimization: Optimization
using OptimizationOptimisers: OptimizationOptimisers
using Random: Random

T = Float32
ArrayType = Array
rng = Random.Xoshiro(123)
ig = 1 # index of the LES grid to use.
include("preprocess_posteriori.jl")

# * Creation of the model: NN closure
closure, θ, st = cnn(;
    setup = setups[ig],
    radii = [3, 3],
    channels = [2, 2],
    activations = [tanh, identity],
    use_bias = [false, false],
    rng
)

# * Define the right hand side of the ODE
dudt_nn2 = create_right_hand_side_with_closure(
    setups[ig], INS.psolver_spectral(setups[ig]), closure, st)

# * Define the loss (a-posteriori) - old way
using Zygote: Zygote
function loss_posteriori(model, p, st, dataloader)
    data = dataloader()
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

NEPOCHS = 500
LR = 1e-2
NUSE_VAL = 2000

# * train a-posteriori: old way single data point
train_data_posteriori = dataloader_posteriori()
optf = Optimization.OptimizationFunction(
    (x, _) -> loss_posteriori(closure, x, st, dataloader_posteriori), # x here is the optimization variable (θ params of NN)
    Optimization.AutoZygote()
)
optprob = Optimization.OptimizationProblem(optf, θ)
optim_result, optim_t, optim_mem, _ = @timed Optimization.solve(
    optprob,
    OptimizationOptimisers.Adam(LR);
    callback = callback,
    maxiters = NEPOCHS,
    progress = true
)
θ_posteriori_optim = optim_result.u

# * package loss
loss_posteriori_lux = create_loss_post_lux(dudt_nn2; sciml_solver = Tsit5())
loss_posteriori_lux(closure, θ, st, train_data_posteriori)

# * training via Lux
lux_result, lux_t, lux_mem, _ = @timed train(
    closure, θ, st, dataloader_posteriori, loss_posteriori_lux;
    nepochs = NEPOCHS, ad_type = Optimization.AutoZygote(),
    alg = OptimizationOptimisers.Adam(LR), cpu = true, callback = callback)

loss, tstate = lux_result
# the trained params are:
θ_posteriori_lux = tstate.parameters

# Function to test the results
function validate_results(p, dataloader, nuse = 100)
    loss = 0
    for i in 1:nuse
        data = dataloader()
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
        loss += T(sum(
            abs2, y[griddims..., :, 1:(size(pred, 4) - 1)] - pred[griddims..., :, 2:end]) /
                  sum(abs2, y))
    end
    return loss / nuse
end
val_optim = validate_results(θ_posteriori_optim, dataloader_posteriori, NUSE_VAL)
val_lux = validate_results(θ_posteriori_lux, dataloader_posteriori, NUSE_VAL)

# Plot the comparison between the two training methods
using Plots
p1 = bar(["Optim", "Lux"], [optim_t, lux_t], ylabel = "Time (s)",
    title = "Training time", legend = false)
p2 = bar(["Optim", "Lux"], [optim_mem, lux_mem], ylabel = "Memory (MB)",
    yaxis = :log, title = "Memory usage", legend = false)
p3 = bar(["Optim", "Lux"], [val_optim, val_lux], ylabel = "Validation",
    yaxis = :log, title = "Validation loss", legend = false)
plot(p1, p2, p3, layout = (3, 1), size = (600, 700),
    title = "A posteriori comparison between loss backends")

# * save the trained model
outdir = "simulations/NavierStokes_2D/outputs"
ispath(outdir) || mkpath(outdir)
@save "$outdir/trained_model_posteriori.jld2" θ_posteriori st
