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
emb_size = 8
patch_size = 3
n_heads = 2
u0 = u

using CoupledNODE: remove_BC
u0 = zeros(T, 512, 512, D, 1)
# load an image 
using TestImages: testimage
u0[:, :, 1, 1] .= testimage("cameraman")
u = remove_BC(u0)
grid = collect(0.0:1.0/509:1.0)
cutoff = 100

using CoupledNODE: create_CNOdownsampler
down_factor = 2
ds = create_CNOdownsampler(T, D, down_factor, cutoff, grid)
size(u)
size(ds(u))
heatmap(ds(u)[:, :, 1, 1], aspect_ratio = 1, title = "ds(u)")

using CoupledNODE: create_CNOupsampler
up_factor = 2
us = create_CNOupsampler(T, D, up_factor, cutoff, grid)
size(u)
size(us(u))
heatmap(us(u)[:, :, 1, 1], aspect_ratio = 1, title = "u")

N = length(grid)
D_up = up_factor * (N - 1) + 1
grid_up = collect(0.0:1.0/(D_up - 1):1.0)
ds2 = create_CNOdownsampler(T, D, down_factor, cutoff, grid_up)
@assert size(ds2(us(u))) == size(u)

D_down = Int(N/down_factor) 
grid_down = collect(0.0:1.0/(D_down - 1):1.0)
us2 = create_CNOupsampler(T, D, up_factor, cutoff, grid_down)
size(us2(ds(u)))
size(u)
fix this one above and add this to the plot
@assert size(us2(ds(u))) == size(u)

# plot side by side u, ds(u), us(u), ds2(us(u))
using Plots: heatmap, plot
p1 = heatmap(u[:, :, 1, 1], aspect_ratio = 1, title = "u")
p2 = heatmap(ds(u)[:, :, 1, 1], aspect_ratio = 1, title = "ds(u)")
p3 = heatmap(us(u)[:, :, 1, 1], aspect_ratio = 1, title = "us(u)")
p4 = heatmap(ds2(us(u))[:, :, 1, 1], aspect_ratio = 1, title = "ds2(us(u))")
plot(p1, p2, p3, p4, layout = (2, 2))

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

# * training via Lux
lux_result, lux_t, lux_mem, _ = @timed train(
    closure, θ, st, dataloader_posteriori, loss_posteriori_lux;
    nepochs = 10, ad_type = Optimization.AutoZygote(),
    alg = OptimizationOptimisers.Adam(0.01), cpu = true, callback = callback)

loss, tstate = lux_result
# the trained params are:
θ_posteriori = tstate.parameters

# * save the trained model
outdir = "simulations/NavierStokes_2D/outputs"
ispath(outdir) || mkpath(outdir)
@save "$outdir/trained_model_posteriori.jld2" θ_posteriori st
