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
params_les = create_params(les_size; nu)
params_dns = create_params(dns_size; nu)
dataseed = 123
dataseed = 2406
data_name = get_data_name(nu, les_size, dns_size, dataseed)
# If they are there load them
if isfile("./simulations/NavierStokes_2D/data/$(data_name).jld2")
    println("Loading data from file")
    simulation_data = load("./simulations/NavierStokes_2D/data/$(data_name).jld2", "data")
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
nuse = 64
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
# Define input channels that do spectral->real
input_channels = (
    u -> real.(ifft(Array(u), (2, 3))),
# notice that the first dimension is the channel, so FT happens on (2,3)
# also, I have to make u into an array cause fft does not work on views
)
# And the output channels that do real->spectral
output_channels = (
    u -> fft(u, (2, 3)),
)
NN_u = create_fno_model(kmax_fno, ch_fno, σ_fno, (params_les.N, params_les.N),
    input_channels = input_channels, output_channels = output_channels)

# Create that cache which is used to compute the right hand side of the Navier-Stokes
cache_les = create_cache(simulation_data.v);
# then we set the NeuralODE model
function F_les(u)
    Zygote.@ignore project(F_NS(u, simulation_data.params_les, cache_les),
        simulation_data.params_les, cache_les)
end
include("./../../../src/NODE.jl")
_closure = create_f_CNODE((F_les,), nothing, (NN_u,); only_closure = true)
# with its parametes
θ, st = Lux.setup(rng, _closure)

# Merge sample and time dimension for a priori fitting
les_data = reshape(simulation_data.v, params_les.N, params_les.N, 2, :)
commutator = reshape(simulation_data.c, params_les.N, params_les.N, 2, :)
# put ch first
les_data = permutedims(les_data, (3, 1, 2, 4))
commutator = permutedims(commutator, (3, 1, 2, 4))
# test closure
x = _closure(les_data, θ, st);

## and the NeuralODE problem
#dt = 2f-4
#tspan = (0.0f0, dt*nunroll)
#prob_neuralode = NeuralODE(_model, tspan, Tsit5(), adaptive=false, dt=dt, saveat=dt)

# Define the loss function
#randloss = create_randloss_derivative(_model, st, simulation_data.v; nuse=nuse)
include("./../../../src/loss_priori.jl")
myloss = create_randloss_derivative(
    les_data,
    commutator,
    _closure,
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
        if l_l % 10 == 0
            plot()
            fig = plot(; xlabel = "Iterations", title = "Loss")
            plot!(fig, 1:10:length(lhist),
                [mean(lhist[i:min(i + 9, length(lhist))]) for i in 1:10:length(lhist)],
                label = "")
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
optf = Optimization.OptimizationFunction((x, p) -> myloss(x), adtype);
optprob = Optimization.OptimizationProblem(optf, pinit);
# And train using Adam + clipping
ClipAdam = OptimiserChain(Adam(1.0f-1), ClipGrad(1));
algo = ClipAdam
using OptimizationCMAEvolutionStrategy, Statistics
algo = CMAEvolutionStrategyOpt();
import OptimizationOptimJL: Optim
algo = Optim.LBFGS();
# ** train loop
result_neuralode = Optimization.solve(
    optprob,
    #ClipAdam;
    algo,
    callback = callback,
    maxiters = 100)

# You can continue the training from here
pinit = result_neuralode.u;
θ = pinit;
optprob = Optimization.OptimizationProblem(optf, pinit);

# Save the results in the TrainedClosure struct
save("trained_models/$(model_name)_$(loss_name)_$(data_name).jld2", "data",
    TrainedNODE(result_neuralode.u, _closure, lhist, model_name, loss_name))
