using CoupledNODE: cnn, create_loss_priori, mean_squared_error, loss_priori_lux,
                   create_stateful_callback, train
using IncompressibleNavierStokes: IncompressibleNavierStokes as INS
using JLD2: @save
using Lux: Lux
using Optimization: Optimization
using OptimizationOptimisers: OptimizationOptimisers
using Random: Random

T = Float32
ArrayType = Array
rng = Random.Xoshiro(123)
ig = 1 # index of the LES grid to use.
include("preprocess_priori.jl")
d = D = setups[ig].grid.dimension()

# * Creation of the model: NN closure
closure, θ, st = cnn(;
    T = T,
    D = D,
    data_ch = D,
    radii = [3, 3],
    channels = [2, 2],
    activations = [tanh, identity],
    use_bias = [false, false],
    rng
)

# Give the CNN a test run
Lux.apply(closure, io_priori[ig].u[:, :, :, 1:1], θ, st)[1]

# * loss a priori
loss_priori = create_loss_priori(mean_squared_error, closure)
# this created function can be called: loss_priori(closure_model, θ, st, data) 
loss_priori(closure, θ, st, train_data_priori) # check that the loss is working

# * loss in the Lux format
loss_priori_lux(closure, θ, st, train_data_priori)

# Define the callback
callbackstate, callback = create_stateful_callback(θ)

# * Training (via Lux)
loss, tstate = train(closure, θ, st, dataloader_prior, loss_priori_lux;
    nepochs = 50, ad_type = Optimization.AutoZygote(),
    alg = OptimizationOptimisers.Adam(0.1), cpu = true, callback = callback)
# the trained parameters at the end of the training are: 
θ_priori = tstate.parameters

# * save the trained model
outdir = "simulations/NavierStokes_2D/outputs"
ispath(outdir) || mkpath(outdir)
@save "$outdir/trained_model_priori.jld2" θ_priori st
