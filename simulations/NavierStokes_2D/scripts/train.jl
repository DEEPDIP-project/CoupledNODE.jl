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
using PreallocationTools
const ArrayType = Array
const solver_algo = Tsit5()
const MY_TYPE = Float32 # use float32 if you plan to use a GPU
import CUDA # Test if CUDA is running
if CUDA.functional()
    CUDA.allowscalar(false)
    const ArrayType = CuArray
    import DiffEqGPU: GPUTsit5
    const solver_algo = GPUTsit5()
end
z = CUDA.functional() ? CUDA.zeros : (s...) -> zeros(MY_TYPE, s...)


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
les_size = 32
dns_size = 64
dataseed = 123
data_name = get_data_name(nu, les_size, dns_size, dataseed)
# If they are there load them
if isfile("./simulations/NavierStokes_2D/data/$(data_name).jld2")
    println("Loading data from file")
    simulation_data = load("./simulations/NavierStokes_2D/data/$(data_name).jld2","data")
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
# Function to generate the model name
function generate_FNO_name(ch_fno, kmax_fno, σ_fno)
    ch_str = join(ch_fno, '-')
    kmax_str = join(kmax_fno, '-')
    σ_str = join(σ_fno, '-')
    
    return "FNO__$(ch_str)__$(kmax_str)__$(σ_str)"
end
model_name = generate_FNO_name(ch_fno, kmax_fno, σ_fno)

# If you want to use a CNN instead, you can use the following:
# 1) radii of the convolutional layers
#r_cnn = [2, 2, 2]
## 2) Number of channels (where the 2-ch at start and end are for the real and imaginary parts)
#ch_cnn = [2, 8, 8, 2]
## 3) Activations
#σ_cnn = [leakyrelu, leakyrelu, identity]
## 4) Bias (use or not)
#b_cnn = [true, true, false]
## and get the corresponding name
#model_name = generate_CNN_name(r_cnn, ch_cnn, σ_cnn, b_cnn)

# Then decide which type of loss function you want between:
# 1) Random derivative (a priori) loss function
# for which we specify how many points to use per epoch
nuse = 50
loss_name = "lossPrior-nu$(nuse)"
## 2) Random trajectory (a posteriori) loss function (DtO)
## for which we need to specify how many steps to unroll per epoch
#nunroll = 10
#loss_name = "lossDtO-nu$(nunroll)"
## 3) DtO multishooting
## for which we also need to specify how many consecutive intervals
#nintervals = 4
#loss_name = "lossMulDtO-nu$(nunroll)-ni$(nintervals)"


# check if the model has been trained already
if isfile("./simulations/NavierStokes_2D/models/$(model_name)_$(loss_name)_$(data_name).jld2")
    throw(DomainError("Model already trained."))
end


## Here we define the closure model
# (for DtO we need single_timestep=true, for derivative fitting we need single_timestep=false)
#_closure = create_cnn_model(r_cnn, ch_cnn, σ_cnn, b_cnn; single_timestep=true)
#_closure = create_fno_model(kmax_fno, ch_fno, σ_fno; single_timestep=true)
include("./../../../src/NS_FNO.jl")
NN_u = create_fno_model(kmax_fno, ch_fno, σ_fno, (les_size, les_size));


# Create that cache which is used to compute the right hand side of the Navier-Stokes
cache_les = create_cache(simulation_data.v);
# then we set the NeuralODE model
F_les(u) = Zygote.@ignore project(F_NS(u, simulation_data.params_les, cache_les), simulation_data.params_les, cache_les)
include("./../../../src/NODE.jl")
_closure = create_f_CNODE((F_les,),nothing, (NN_u,); only_closure = true)
# with its parametes
θ, st = Lux.setup(rng, _closure)

les_data = reshape(simulation_data.v, les_size, les_size, 2, :)
# put ch first
les_data = permutedims(les_data, (3, 1, 2, 4))
_closure(les_data, θ, st)
## and the NeuralODE problem
#dt = 2f-4
#tspan = (0.0f0, dt*nunroll)
#prob_neuralode = NeuralODE(_model, tspan, Tsit5(), adaptive=false, dt=dt, saveat=dt)

# Define the loss function
#randloss = create_randloss_derivative(_model, st, simulation_data.v; nuse=nuse)
include("./../../../src/loss_priori.jl")
# to define the a priori loss function I need to batch the data 
u = simulation_data.u
u[1]

pp = spectral_cutoff(u[1], simulation_data.params_les.K)
_model(pp, θ, st)
myloss = create_randloss_derivative(
    simulation_data.v,
    simulation_data.c,
    _model,
    st;
    n_use = nuse,
    λ = 0,
    λ_c = 0);
#randloss = create_randloss_DtO(simulation_data.v)
#randloss = create_randloss_MulDtO(simulation_data.v)
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
callback(pinit, myloss(pinit)...)


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
