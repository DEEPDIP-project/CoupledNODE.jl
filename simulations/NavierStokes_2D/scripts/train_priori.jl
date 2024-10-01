using Random: Random
using IncompressibleNavierStokes: IncompressibleNavierStokes as INS

T = Float32
ArrayType = Array
rng = Random.Xoshiro(123)
ig = 1 # index of the LES grid to use.
include("preprocess_priori.jl")

# * Creation of the model: NN closure
using CoupledNODE: cnn
closure, θ, st = cnn(;
    setup = setups[ig],
    radii = [3, 3],
    channels = [2, 2],
    activations = [tanh, identity],
    use_bias = [false, false],
    rng
)

# Give the CNN a test run
using Lux: Lux
Lux.apply(closure, io_priori[ig].u[:, :, :, 1:1], θ, st)[1]

# * loss a priori
using CoupledNODE: create_loss_priori, mean_squared_error
loss_priori = create_loss_priori(mean_squared_error, closure)
# this created function can be called: loss_priori(closure_model, θ, st, data) 
loss_priori(closure, θ, st, train_data_priori) # check that the loss is working

# * loss in the Lux format
using CoupledNODE: loss_priori_lux
loss_priori_lux(closure, θ, st, train_data_priori)

# * old way of training
using CoupledNODE: callback
using Optimization: Optimization
using OptimizationOptimisers: OptimizationOptimisers
optf = Optimization.OptimizationFunction(
    (u, _) -> loss_priori(closure, u, st, train_data_priori), # u here is the optimization variable (θ params of NN)
    Optimization.AutoZygote())
optprob = Optimization.OptimizationProblem(optf, θ)
result_priori = Optimization.solve(
    optprob,
    OptimizationOptimisers.Adam(0.1);
    callback = callback,
    maxiters = 50,
    progress = true
)
θ_priori = result_priori.u
# with this approach, we have the problem that we cannot loop trough the data. 

# another option of callback
using CoupledNODE: create_stateful_callback
callbackstate, callback_2 = create_stateful_callback(θ)

# * new way of training (via Lux)
using CoupledNODE: train
loss, tstate = train(closure, θ, st, dataloader_prior, loss_priori_lux;
    nepochs = 50, ad_type = Optimization.AutoZygote(),
    alg = OptimizationOptimisers.Adam(0.1), cpu = true, callback = callback_2)
# the trained parameters at the end of the training are: 
θ_priori = tstate.parameters
