using CoupledNODE: cnn, train, create_loss_post_lux, create_callback
using CoupledNODE.NavierStokes: create_right_hand_side_with_closure
using DifferentialEquations: ODEProblem, solve, Tsit5
using IncompressibleNavierStokes: IncompressibleNavierStokes as INS
using JLD2: @save
using Optimization: Optimization
using OptimizationOptimisers: OptimizationOptimisers
using Random: Random

T = Float32
rng = Random.Xoshiro(123)
ig = 1 # index of the LES grid to use.
include("preprocess_posteriori.jl")

using ComponentArrays: ComponentArray
using Lux: Lux
u = io_post[ig].u[:, :, :, 1, 1:50]
T = setups[1].T
d = D = setups[1].grid.dimension()
u0 = u

#************88
# Test that the downsampler and upsampler work
using TestImages: testimage
using Plots: heatmap, plot, plot!


N0 = 512
u0 = zeros(T, N0, N0, D, 1)
u0[:, :, 1, 1] = testimage("cameraman")
typeof(u0)
cutoff = 0.1

using CoupledNODE: create_CNO
ch_ = [2,2]
df = [2,2]
k_rad = [3,2]
bd = [5,5]
model = create_CNO(T=T, N=N0, D=D, cutoff=cutoff, ch_sizes=ch_, down_factors=df, k_radii=k_rad, bottleneck_depths = bd)
θ, st = Lux.setup(rng, model)
using ComponentArrays: ComponentArray
θ = ComponentArray(θ)
heatmap(model(u0, θ, st)[1][:, :, 1, 1], aspect_ratio = 1, title = "model(u0)")


using Zygote: Zygote
model(u0, θ, st)[1] .- u0
loss(θ) = sum((model(u0, θ, st)[1][:,:,1,1] .- u0[:,:,1,1])^2)
loss(θ)
g = Zygote.gradient(θ->loss(θ), θ)

# test training with optimize
using Optimization: Optimization
optf = Optimization.OptimizationFunction(
    (x, _) -> loss(x),
    Optimization.AutoZygote()
)
optprob = Optimization.OptimizationProblem(optf, θ)
optim_result, optim_t, optim_mem, _ = @timed Optimization.solve(
    optprob,
    OptimizationOptimisers.Adam(0.01);
    maxiters = 20,
    progress = true
)
θ_p = optim_result.u
heatmap(model(u0, θ_p, st)[1][:, :, 1, 1], aspect_ratio = 1, title = "model(u0)")
θ = θ_p

#TODO use new callback (merge main!)


#############



# * Define the right hand side of the ODE
dudt_nn2 = create_right_hand_side_with_closure(
    setups[ig], INS.psolver_spectral(setups[ig]), closure, st)

# * Define the loss (a-posteriori) 
train_data_posteriori = dataloader_posteriori()
loss_posteriori_lux = create_loss_post_lux(dudt_nn2; sciml_solver = Tsit5())
loss_posteriori_lux(closure, θ, st, train_data_posteriori)

# * Callback function
callbackstate_val, callback_val = create_callback(
    dudt_nn2, θ, test_io_post[ig], loss_posteriori_lux, st, nunroll = 3 * nunroll,
    rng = rng, do_plot = true, plot_train = false)
θ_posteriori = θ

# * training via Lux
lux_result, lux_t, lux_mem, _ = @timed train(
    closure, θ_posteriori, st, dataloader_posteriori, loss_posteriori_lux;
    nepochs = 50, ad_type = Optimization.AutoZygote(),
    alg = OptimizationOptimisers.Adam(0.01), cpu = true, callback = callback_val)

loss, tstate = lux_result
# the trained params are:
θ_posteriori = tstate.parameters

# * save the trained model
outdir = "simulations/NavierStokes_2D/outputs"
ispath(outdir) || mkpath(outdir)
@save "$outdir/trained_model_posteriori.jld2" θ_posteriori st
