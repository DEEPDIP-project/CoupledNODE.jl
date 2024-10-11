using CoupledNODE: cnn, train, create_loss_post_lux
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

# * Define the loss (a-posteriori) 
train_data_posteriori = dataloader_posteriori()
loss_posteriori_lux = create_loss_post_lux(dudt_nn2; sciml_solver = Tsit5())
loss_posteriori_lux(closure, θ, st, train_data_posteriori)

# * Callback function
using CoupledNODE: create_callback
callback_validation = create_callback(
    dudt_nn2, test_io_post, nunroll = 3 * nunroll, rng = rng, plot_train = false)
θ_posteriori = θ

# * training via Lux
lux_result, lux_t, lux_mem, _ = @timed train(
    closure, θ_posteriori, st, dataloader_posteriori, loss_posteriori_lux;
    nepochs = 1000, ad_type = Optimization.AutoZygote(),
    alg = OptimizationOptimisers.Adam(0.01), cpu = true, callback = callback_validation)

loss, tstate = lux_result
# the trained params are:
θ_posteriori = tstate.parameters

# * save the trained model
outdir = "simulations/NavierStokes_2D/outputs"
ispath(outdir) || mkpath(outdir)
@save "$outdir/trained_model_posteriori.jld2" θ_posteriori st
