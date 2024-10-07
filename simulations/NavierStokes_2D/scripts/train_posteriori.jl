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

using CoupledNODE: AttentionLayer
using ComponentArrays: ComponentArray
using Lux: Lux
u = io_post[ig].u[:, :, :, 1, 1:50]
T = setups[1].T
d = D = setups[1].grid.dimension()
N = size(u, 1)
emb_size = 8
patch_size = 3
n_heads = 2

# * Define the CNN layers
# since I will use them after the attention (that gets concatenated with the input), I have to start from 2*D channels
CnnLayers, _, _ = cnn(;
    T = T,
    D = D,
    data_ch = 2 * D,
    radii = [3, 3],
    channels = [2, 2],
    activations = [tanh, identity],
    use_bias = [false, false],
    rng
)
layers = (
    Lux.SkipConnection(AttentionLayer(N, d, emb_size, patch_size, n_heads; T = T),
        (x, y) -> cat(x, y; dims = 3); name = "Attention"),
    CnnLayers
)
closure = Lux.Chain(layers...)
θ, st = Lux.setup(rng, closure)
using ComponentArrays: ComponentArray
θ = ComponentArray(θ)

# test and trigger the model
closure(u, θ, st)

# * Define the right hand side of the ODE
dudt_nn2 = create_right_hand_side_with_closure(
    setups[ig], INS.psolver_spectral(setups[ig]), closure, st)

# * Define the loss (a-posteriori) - old way
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
    maxiters = 100,
    progress = true
)
θ_posteriori = result_posteriori.u

# * package loss
loss_posteriori_lux = create_loss_post_lux(dudt_nn2; sciml_solver = Tsit5())
loss_posteriori_lux(closure, θ, st, train_data_posteriori)

# * training via Lux
loss, tstate = train(closure, θ, st, dataloader_posteriori, loss_posteriori_lux;
    nepochs = 100, ad_type = Optimization.AutoZygote(),
    alg = OptimizationOptimisers.Adam(0.1), cpu = true, callback = callback)

# the trained params are:
θ_posteriori = tstate.parameters

# * save the trained model
outdir = "simulations/NavierStokes_2D/outputs"
ispath(outdir) || mkpath(outdir)
@save "$outdir/trained_model_posteriori.jld2" θ_posteriori st
