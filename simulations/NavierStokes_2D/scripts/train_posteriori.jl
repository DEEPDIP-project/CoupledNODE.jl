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

# downsize the input which would be too large to autodiff
using CoupledNODE: create_CNOdownsampler
down_factor = 2
ds = create_CNOdownsampler(T, D, N0, down_factor, cutoff)
u = ds(u0)
N = size(u, 1)


using CoupledNODE: CNODownsampleBlock
down_factor = 2
dl = CNODownsampleBlock(N, D, down_factor, 2, 1, use_batchnorm =false, T=T)

using CoupledNODE: CNOUpsampleBlock
up_factor = 2
dl = CNOUpsampleBlock(N, D, up_factor, 2, 1, use_batchnorm =false, T=T)

using CoupledNODE: create_CNO
cn = create_CNO(T, N, D, channels= [2, 2], down_factors = [2, 2], up_factors = [2, 2], cutoffs=[0.1, 0.1], use_batchnorms=[false, false], use_biases=[false, false], conv_per_block=1) 



# ***** Test if you can differentiate upsample and downsample
layers = dl.layers
layers = cn
closure = Lux.Chain(layers...)
θ, st = Lux.setup(rng, closure)
using ComponentArrays: ComponentArray
θ = ComponentArray(θ)

# test and trigger the model
closure(u, θ, st)
heatmap(closure(u, θ, st)[1][:, :, 1, 1], aspect_ratio = 1, title = "closure(u)")
using Zygote: Zygote
Zygote.gradient(θ -> closure(u, θ, st)[1][1], θ)


#############
assemble the blocks: where do the parameter go?


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
