using CoupledNODE: cnn, create_loss_priori, mean_squared_error, loss_priori_lux,
                   create_callback, train
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
else
    @info "Running on CPU"
    dev = Lux.cpu_device()
    cpu = true
end

rng = Random.Xoshiro(123)
ig = 1 # index of the LES grid to use.
include("preprocess_priori.jl")
d = D = setups[ig].grid.dimension()

# * Creation of the model: NN closure
closure, θ, st = cnn(;
    T = T,
    D = params.D,
    data_ch = params.D,
    radii = [2, 2, 2, 2],
    channels = [8, 8, 8, 2],
    activations = [tanh, tanh, tanh, identity],
    use_bias = [true, true, true, false],
    rng
)
θ, st = (θ, st) .|> dev # move to gpu if available
# Give the CNN a test run
Lux.apply(closure, dev(io_priori[ig].u[:, :, :, 1:1]), θ, st)[1]

# * loss in the Lux format
loss_priori_lux(closure, θ, st, dev(train_data_priori))

# * Define the callback
callbackstate_val, callback_val = create_callback(
    closure, θ, test_io_post[ig], loss_priori_lux, st, batch_size = 64,
    rng = rng, do_plot = false, plot_train = false, device = dev)

# * Training (via Lux)
opt = ClipAdam = OptimiserChain(Adam(T(1.0e-2)), ClipGrad(1));
loss, tstate = train(closure, θ, st, dataloader_prior, loss_priori_lux;
    nepochs = 50, ad_type = Optimization.AutoZygote(),
    alg = opt, cpu = cpu, callback = callback_val)
# the trained parameters at the end of the training are: 
θ_priori = tstate.parameters

# * save the trained model
outdir = "simulations/NavierStokes_2D/outputs"
ispath(outdir) || mkpath(outdir)
@save "$outdir/trained_model_priori.jld2" θ_priori st
