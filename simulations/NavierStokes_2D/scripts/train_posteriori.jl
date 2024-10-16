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
grid = collect(0.0:1.0/(N0-1):1.0)
cutoff = 5

# downsize the input which would be too large to autodiff
using CoupledNODE: create_CNOdownsampler
down_factor = 6
ds = create_CNOdownsampler(T, D, down_factor, cutoff, grid)
u = ds(u0)
N = size(u, 1)
grid = collect(0.0:1.0/(N-1):1.0)
heatmap(u[:, :, 1, 1], aspect_ratio = 1, title = "u")

# define a downsampling op
down_factor = 2
ds = create_CNOdownsampler(T, D, down_factor, cutoff, grid)

# define an upsampling op
using CoupledNODE: create_CNOupsampler
up_factor = 2
us = create_CNOupsampler(T, D, up_factor, cutoff, grid)

# test upsampling->downsampling
D_up = up_factor * N
grid_up = collect(0.0:1.0/(D_up - 1):1.0)
ds2 = create_CNOdownsampler(T, D, down_factor, cutoff, grid_up)
@assert size(ds2(us(u))) == size(u)

# test downsampling->upsampling
D_down = Int(N/down_factor) 
grid_down = collect(0.0:1.0/(D_down - 1):1.0)
us2 = create_CNOupsampler(T, D, up_factor, cutoff, grid_down)
@assert size(us2(ds(u))) == size(u)

# summary plot
p1 = heatmap(u[:, :, 1, 1], aspect_ratio = 1, title = "u", colorbar=false)
p2 = heatmap(ds(u)[:, :, 1, 1], aspect_ratio = 1, title = "ds(u)", colorbar=false)
p3 = heatmap(us(u)[:, :, 1, 1], aspect_ratio = 1, title = "us(u)", colorbar=false)
p4 = heatmap(ds2(us(u))[:, :, 1, 1], aspect_ratio = 1, title = "ds2(us(u))", colorbar=false)
p5 = heatmap(us2(ds(u))[:, :, 1, 1], aspect_ratio = 1, title = "us2(ds(u))", colorbar=false)
plot(p1, p2, p3, p4, p5, layout = (2, 3))


# ***** Test if you can differentiate upsample and downsample

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
ds = create_CNOdownsampler(T, D, down_factor, cutoff, grid)
us = create_CNOupsampler(T, D, up_factor, cutoff, grid_down)
layers = (
    x -> ds(x),
    x -> us(x),
    #CnnLayers
)
closure = Lux.Chain(layers...)
θ, st = Lux.setup(rng, closure)
using ComponentArrays: ComponentArray
θ = ComponentArray(θ)

# test and trigger the model
using FFTW: FFTW
Zygote.@adjoint FFTW.fft(xs) = (FFTW.fft(xs), (Δ)-> (FFTW.ifft(Δ),))
Zygote.@adjoint FFTW.ifft(xs) = (FFTW.ifft(xs), (Δ)-> (FFTW.fft(Δ),))
closure(u, θ, st)
heatmap(closure(u, θ, st)[1][:, :, 1, 1], aspect_ratio = 1, title = "closure(u)")
using Zygote: Zygote
Zygote.gradient(θ -> closure(u, θ, st)[1][1], θ)


#############


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
