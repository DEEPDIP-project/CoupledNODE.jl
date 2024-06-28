using ComponentArrays
using CUDA
using FFTW
using LinearAlgebra
using Lux
using LuxCUDA
using NNlib
using Optimisers
using Plots
using Printf
using Random
using Zygote
using DifferentialEquations
using JLD2
using SciMLSensitivity
using DiffEqFlux
using OptimizationOptimisers
using Statistics

include("./../../../src/NavierStokes.jl")
include("./../../../src/NODE.jl")


# Lux likes to toss random number generators around, for reproducible science
rng = Random.default_rng()

# This line makes sure that we don't do accidental CPU stuff while things
# should be on the GPU
CUDA.allowscalar(false)

# fix the random seed for reproducibility
Random.seed!(1234)

# Select the parameters that define the simulation you want to target
nu = 5.0f-4
les_size = 64
dns_size = 128
#les_size = 32
#dns_size = 64
dataseed = 1234
data_name = get_data_name(nu, les_size, dns_size, dataseed)
# If they are there load them
if isfile("./simulations/NavierStokes_2D/data/$(data_name).jld2")
    println("Loading data from file")
    simulation_data = load("data/$(data_name).jld2","data")
else
    throw(DomainError("Data are missing. You may want to generate them first."))
end


# ### Model architecture

# We will use a FNO, but let's specify the architecture here:
# 1) Number of channels (where the 2-ch at start and end are for the real and imaginary parts)
ch_fno = [2, 5, 5, 5, 2]
# 2) Cut-off wavenumbers
kmax_fno = [8, 8, 8, 8]
# 3) Activations
σ_fno = [gelu, gelu, gelu, identity]
# We will identity this architecture with the name 
model_name = generate_FNO_name(ch_fno, kmax_fno, σ_fno)

# If you want to use a CNN instead, you can use the following:
# 1) radii of the convolutional layers
r_cnn = [2, 2, 2]
# 2) Number of channels (where the 2-ch at start and end are for the real and imaginary parts)
ch_cnn = [2, 8, 8, 2]
# 3) Activations
σ_cnn = [leakyrelu, leakyrelu, identity]
# 4) Bias (use or not)
b_cnn = [true, true, false]
# and get the corresponding name
model_name = generate_CNN_name(r_cnn, ch_cnn, σ_cnn, b_cnn)

# Then decide which type of loss function you want between:
# 1) Random derivative (a priori) loss function
# for which we specify how many points to use per epoch
nuse = 50
loss_name = "lossPrior-nu$(nuse)"
# 2) Random trajectory (a posteriori) loss function (DtO)
# for which we need to specify how many steps to unroll per epoch
nunroll = 10
loss_name = "lossDtO-nu$(nunroll)"
# 3) DtO multishooting
# for which we also need to specify how many consecutive intervals
nintervals = 4
loss_name = "lossMulDtO-nu$(nunroll)-ni$(nintervals)"


# check if the model has been trained already
if isfile("./simulations/NavierStokes_2D/models/$(model_name)_$(loss_name)_$(data_name).jld2")
    throw(DomainError("Model already trained."))
end


## Here we define the closure model
# (for DtO we need single_timestep=true, for derivative fitting we need single_timestep=false)
_closure = create_cnn_model(r_cnn, ch_cnn, σ_cnn, b_cnn; single_timestep=true)
_closure = create_fno_model(kmax_fno, ch_fno, σ_fno; single_timestep=true)

# then we set the NeuralODE model
_model = create_node(_closure, simulation_data.params_les; is_closed=true)
# with its parametes
θ, st = Lux.setup(rng, _model)


# and the NeuralODE problem
dt = 2f-4
tspan = (0.0f0, convert(Float32,dt*nunroll))
prob_neuralode = NeuralODE(_model, tspan, Tsit5(), adaptive=false, dt=dt, saveat=dt)

# Define the loss function
randloss = create_randloss_derivative(_model, st, simulation_data.v; nuse=nuse)
randloss = create_randloss_DtO(simulation_data.v)
randloss = create_randloss_MulDtO(simulation_data.v)
lhist = Float32[]

# Define a callback function to observe training
callback = function (p, l, pred; doplot = true)
    l_l = length(lhist)
    println("Loss[$(l_l)]: $(l)")
    push!(lhist, l)
    if doplot
        # plot rolling average of loss, every 10 steps
        if l_l%10 == 0
            plot()
            fig = plot(; xlabel = "Iterations", title = "Loss")
            plot!(fig, 1:10:length(lhist), [mean(lhist[i:min(i+9, length(lhist))]) for i in 1:10:length(lhist)], label = "")
            display(fig)
        end
    end
    return false
end

# Initialize and trigger the compilation of the model
pinit = ComponentArray(θ);
callback(pinit, randloss(pinit)...)


# Select the autodifferentiation type
adtype = Optimization.AutoZygote()
# We transform the NeuralODE into an optimization problem
optf = Optimization.OptimizationFunction((x, p) -> randloss(x), adtype);
optprob = Optimization.OptimizationProblem(optf, pinit);
# And train using Adam + clipping
ClipAdam = OptimiserChain(Adam(1.0f-2), ClipGrad(1));
# ** train loop
result_neuralode = Optimization.solve(optprob,
    ClipAdam;
    callback = callback,
    maxiters = 10)

# You can continue the training from here
pinit = result_neuralode.u

# Save the results in the TrainedClosure struct
save("trained_models/$(model_name)_$(loss_name)_$(data_name).jld2", "data", TrainedNODE(result_neuralode.u, _closure, lhist, model_name, loss_name))
