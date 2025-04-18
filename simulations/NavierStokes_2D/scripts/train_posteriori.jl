using CoupledNODE: cnn, train, create_loss_post_lux, create_callback
using CoupledNODE.NavierStokes: create_right_hand_side_with_closure
using DifferentialEquations: Tsit5
using IncompressibleNavierStokes: IncompressibleNavierStokes as INS
using JLD2: @save
using Lux: Lux
using Optimization: Optimization
using OptimizationOptimisers: OptimizationOptimisers, Adam, ClipGrad, OptimiserChain
using Random: Random

T = Float32
# Use GPU if available
using CUDA: CUDA
if CUDA.functional()
    @info "Running on CUDA"
    using LuxCUDA
    CUDA.allowscalar(false)
    dev = Lux.gpu_device()
    cpu = false
    backend = CUDA.CUDABackend()
else
    @info "Running on CPU"
    dev = Lux.cpu_device()
    cpu = true
    backend = INS.CPU()
end

rng = Random.Xoshiro(123)
ig = 1 # index of the LES grid to use.
nunroll = 3
include("preprocess_posteriori.jl")

using ComponentArrays: ComponentArray
using Lux: Lux
u = io_post[ig].u[:, :, :, 1, 1:50]
#T = setups[1].T
d = D = setups[1].grid.dimension()
N = size(u, 1)

# * Define the CNN layers
closure, θ,
st = cnn(;
    T = T,
    D = params.D,
    data_ch = params.D,
    radii = [2, 2, 2, 2],
    channels = [8, 8, 8, 2],
    activations = [tanh, tanh, tanh, identity],
    use_bias = [true, true, true, false],
    rng
)

# test and trigger the model
θ, st = (θ, st) .|> dev # move to gpu if available
closure(dev(u), θ, st)

# * Define the right hand side of the ODE
dudt_nn2 = create_right_hand_side_with_closure(
    setups[ig], INS.psolver_spectral(setups[ig]), closure, st)

# * Define the loss (a-posteriori)
train_data_posteriori = dataloader_posteriori()
loss_posteriori_lux = create_loss_post_lux(dudt_nn2; sciml_solver = Tsit5(), cpu = cpu)
loss_posteriori_lux(closure, θ, st, dev(train_data_posteriori))

# * Callback function
callbackstate_val,
callback_val = create_callback(
    dudt_nn2, θ, test_io_post[ig], loss_posteriori_lux, st, nunroll = 2 * nunroll,
    rng = rng, do_plot = false, plot_train = false, device = dev)
θ_posteriori = θ

# * training via Lux
opt = ClipAdam = OptimiserChain(Adam(T(1.0e-2)), ClipGrad(1));
lux_result, lux_t,
lux_mem,
_ = @timed train(
    closure, θ_posteriori, st, dataloader_posteriori, loss_posteriori_lux;
    nepochs = 300, ad_type = Optimization.AutoZygote(),
    alg = opt, cpu = true, callback = callback_val)

loss, tstate = lux_result
# the trained params are:
θ_posteriori = tstate.parameters

# * save the trained model
outdir = "simulations/NavierStokes_2D/outputs"
ispath(outdir) || mkpath(outdir)
@save "$outdir/trained_model_posteriori.jld2" θ_posteriori st
