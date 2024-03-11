using Lux
using SciMLSensitivity
using DiffEqFlux
using DifferentialEquations
using Plots
using Plots.PlotMeasures
using Zygote
using Random
rng = Random.seed!(1234)
using OptimizationOptimisers
using Statistics
using ComponentArrays
using CUDA
using Images
using Interpolations
using NNlib
ArrayType = CUDA.functional() ? CuArray : Array;
## Import our custom backend functions
include("coupling_functions/functions_example.jl")
include("coupling_functions/functions_NODE.jl")
include("coupling_functions/functions_loss.jl")
include("coupling_functions/functions_FDderivatives.jl");
include("coupling_functions/functions_CNODE_loss.jl")

# ## Learning the Gray-Scott model

# We have seen in the following example how to use the CNODE to solve the Gray-Scott model.
# Here we want to introduce the Learning part of the CNODE, and show how it can be used to close the CNODE.

# As a reminder, the GS model is defined from 
# \begin{equation}\begin{cases} \frac{du}{dt} = D_u \nabla u - uv^2 + f(1-u)  \equiv F_u(u,v) \\ \frac{dv}{dt} = D_v \nabla v + uv^2 - (f+k)v  \equiv G_v(u,v)\end{cases} \end{equation}
# where $u(x,y,t):\mathbb{R}^2\times \mathbb{R}\rightarrow \mathbb{R}$ is the concentration of species 1, while $v(x,y,t)$ is the concentration of species two. This model reproduce the effect of the two species diffusing in their environment, and reacting together.
# This effect is captured by the ratios between $D_u$ and $D_v$ (the diffusion coefficients) and $f$ and $k$ (the reaction rates).

# In this example, we will first use the exact GS model to gather some data
# then in the second part, we will train a generic CNODE to show that it can learn the GS model from the data.

# ### Solving GS to collect data
# Definition of the grid
dux = duy = dvx = dvy = 1.
nux = nuy = nvx = nvy = 64
grid = Grid(dux, duy, nux, nuy, dvx, dvy, nvx, nvy);


# Here, we define the initial condition as a random perturbation over a constant background to add variety
function initial_condition(grid, U₀, V₀, ε_u, ε_v; nsimulations=1)
    u_init = U₀ .+ ε_u .* randn(grid.nux, grid.nuy, nsimulations)
    v_init = V₀ .+ ε_v .* randn(grid.nvx, grid.nvy, nsimulations)
    return u_init, v_init
end
U₀ = 0.5    # initial concentration of u
V₀ = 0.25   # initial concentration of v
ε_u = 0.05 # magnitude of the perturbation on u
ε_v = 0.1 # magnitude of the perturbation on v
u_initial, v_initial = initial_condition(grid, U₀, V₀, ε_u, ε_v, nsimulations=10);

# We can now define the initial condition as a flattened concatenated array
uv0 = vcat(reshape(u_initial, grid.Nu,:),reshape(v_initial, grid.Nv,:));

# These are the GS parameters that we will later try to learn
D_u = 0.16
D_v = 0.08
f = 0.055
k = 0.062;

# Exact right hand side of the GS model
F_u(u,v,grid) = D_u*Laplacian(u,grid.dux,grid.duy) .- u.*v.^2 .+ f.*(1.0.-u)
G_v(u,v,grid) = D_v*Laplacian(v,grid.dvx,grid.dvy) .+ u.*v.^2 .- (f+k).*v

# CNODE definition
f_CNODE = create_f_CNODE(F_u, G_v, grid; is_closed=false);
θ, st = Lux.setup(rng, f_CNODE);

# Burnout run
trange_burn = (0.0f0, 1.0f0)
dt, saveat = (1e-2, 1)
burnout_CNODE = NeuralODE(f_CNODE, trange_burn, Tsit5(), adaptive=false, dt=dt, saveat=saveat);
burnout_CNODE_solution = Array(burnout_CNODE(uv0, θ, st)[1]);
# Second burnout with larger timestep
trange_burn = (0.0f0, 500.0f0)
dt, saveat = (1/(4*max(D_u,D_v)), 100)
burnout_CNODE = NeuralODE(f_CNODE, trange_burn, Tsit5(), adaptive=false, dt=dt, saveat=saveat);
burnout_CNODE_solution = Array(burnout_CNODE(burnout_CNODE_solution[:,:,end], θ, st)[1]);

# Data collection run
uv0 = burnout_CNODE_solution[:,:,end];
trange = (0.0f0, 7000.0f0)
trange = (0.0f0, 2000.0f0)
dt, saveat = (1/(4*max(D_u,D_v)), 1)
GS_CNODE = NeuralODE(f_CNODE, trange, Tsit5(), adaptive=false, dt=dt, saveat=saveat);
GS_data = Array(GS_CNODE(uv0, θ, st)[1])
u = reshape(GS_data[1:grid.Nu, :, :]    , grid.nux, grid.nuy, size(GS_data,2), :);
v = reshape(GS_data[grid.Nu+1:end, :, :], grid.nvx, grid.nvy, size(GS_data,2), :);


# ### Training a CNODE to learn the GS model
# To learn the GS model, we will use the following CNODE
# \begin{equation}\begin{cases} \frac{du}{dt} = D_u \nabla u + \theta_{u,1} uv^2 +\theta_{u,2} v^2u + \theta_{u,3} u +\theta_{u,4} v +\theta_{u,5}  \\ \frac{dv}{dt} = D_v \nabla v + \theta_{v,1} uv^2 + \theta_{v,2} v^2u +\theta_{v,3} u +\theta_{v,4} v +\theta_{v,5} \end{cases} \end{equation}
# So in this example the deterministic force is only the diffusion, while the model has to learn the interaction between the fields and the source terms.
# Then the deterministic force is
F_u(u,v,grid) = Zygote.@ignore D_u*Laplacian(u,grid.dux,grid.duy) 
G_v(u,v,grid) = Zygote.@ignore D_v*Laplacian(v,grid.dvx,grid.dvy) 
# where we are telling Zygote to ignore this tree branch for the gradient propagation.


# For the trainable part, we define an abstract Lux layer
struct GSLayer{F} <: Lux.AbstractExplicitLayer
    init_weight::F
end
# and its costructor
function GSLayer(; init_weight = glorot_uniform)
    return GSLayer(init_weight)
end
# We also need to specify how to initialize its parameters and states. 
# This custom layer does not have any hidden states (RNGs) that are modified.
Lux.initialparameters(rng::AbstractRNG, (; init_weight)::GSLayer) = (;
    gs_weights = init_weight(rng, 5),
)
Lux.initialstates(::AbstractRNG, ::GSLayer) = (;)
Lux.parameterlength((; )::GSLayer) = 5
Lux.statelength(::GSLayer) = 0

# We now define how to pass inputs through GSlayer, assuming the
# following:
# - Input size: `(N, N, 2, nsample)`, where the two channel are u and v
# - Output size: `(N, N, nsample)` where we assumed monochannel output, so we dropped the channel dimension

# This is what each layer does:
function ((;)::GSLayer)(x, params, state)
    N = size(x, 1)
    u = x[:,:,1,:]
    v = x[:,:,2,:]

    out = params.gs_weights[1] .* u .* u .* v .+ params.gs_weights[2] .* u .* v .* v .+ params.gs_weights[3] .* u .+ params.gs_weights[4] .* v .+ params.gs_weights[5]

    ## The layer does not modify state
    out, state
end
# Function to create the model. In this case is just a GS layer wrapped in a Chain, but the model can be as complex as needed.
function create_GS_model()
    return Chain(
        GSLayer(),
    )
end

# Here we define the trainable part
NN_u = create_GS_model()
NN_v = create_GS_model()

# We can now close the CNODE with the Neural Network
f_closed_CNODE = create_f_CNODE(F_u, G_v, grid, NN_u, NN_v; is_closed=true) 
θ, st = Lux.setup(rng, f_closed_CNODE);
print(θ)

# ### Design the **loss function**
# For this example, we use *multishooting a posteriori* fitting (MulDtO), where we tell `Zygote` to compare `nintervals` of length `nunroll` to get the gradient. Notice that this method is differentiating through the solution of the NODE!
nunroll = 20   
nintervals = 5
nsamples = 3
# We also define this auxiliary NODE that will be used for training
# We can use smaller time steps for the training 
dt_train = 1
# but we have to sample at the same rate as the data
saveat_train = saveat
t_train_range = (0.0f0, saveat_train*(nunroll+0)) # it has to be as long as unroll
training_CNODE = NeuralODE(f_closed_CNODE, t_train_range, Tsit5(), adaptive=false, dt=dt_train, saveat=saveat_train);

# * Create the loss
myloss = create_randloss_MulDtO(GS_data, nunroll=nunroll, nintervals=nintervals, nsamples=nsamples, λ=0);

# To initialize the training, we need some objects to monitor the procedure, and we trigger the first compilation.

lhist = Float32[];
# Initialize and trigger the compilation of the model
pinit = ComponentArray(θ)
myloss(pinit)  # trigger compilation
# [!] Check that the loss does not get type warnings, otherwise it will be slower


# Select the autodifferentiation type
adtype = Optimization.AutoZygote();
# We transform the NeuralODE into an optimization problem
optf = Optimization.OptimizationFunction((x, p) -> myloss(x), adtype);
optprob = Optimization.OptimizationProblem(optf, pinit);

# Select the training algorithm:
ClipAdam = OptimiserChain(Adam(1.0f-3), ClipGrad(1));

# Finally we can train the NODE
result_neuralode = Optimization.solve(optprob,
    ClipAdam;
    callback = callback,
    maxiters = 50
    );
pinit = result_neuralode.u;
optprob = Optimization.OptimizationProblem(optf, pinit);
# (Notice that the block above can be repeated to continue training)


# and finally I use the trained CNODE to compare the solution with the target
trange = (0.0f0, saveat*nunroll)
trained_CNODE = NeuralODE(f_closed_CNODE, trange, Tsit5(), adaptive=false, dt=dt, saveat=saveat);
trained_CNODE_solution = Array(trained_CNODE(GS_data[:,1:3,1], θ, st)[1]);
u_trained = reshape(trained_CNODE_solution[1:grid.Nu, :, :]    , grid.nux, grid.nuy, size(trained_CNODE_solution,2), :);
v_trained = reshape(trained_CNODE_solution[grid.Nu+1:end, :, :], grid.nvx, grid.nvy, size(trained_CNODE_solution,2), :);
anim = Animation()
fig = plot(layout = (2, 5), size = (750, 300))
@gif for i in 1:1:size(u_trained, 4)
    p1 = heatmap(u[:,:,1,i], axis=false, cbar=false, aspect_ratio=1, color=:reds, title="Exact" )
    p2 = heatmap(v[:,:,1,i], axis=false, cbar=false, aspect_ratio=1, color=:blues)
    p3 = heatmap(u_trained[:,:,1,i], axis=false, cbar=false, aspect_ratio=1, color=:reds, title="Trained")
    p4 = heatmap(v_trained[:,:,1,i], axis=false, cbar=false, aspect_ratio=1, color=:blues)
    et = abs.(u[:,:,1,i] .- u_trained[:,:,1,i])
    p5 = heatmap(et, axis=false, cbar=false, aspect_ratio=1, color=:greens, title = "Diff-u")
    p6 = heatmap(u[:,:,2,i], axis=false, cbar=false, aspect_ratio=1, color=:reds  )
    p7 = heatmap(v[:,:,2,i], axis=false, cbar=false, aspect_ratio=1, color=:blues )
    p8 = heatmap(u_trained[:,:,2,i], axis=false, cbar=false, aspect_ratio=1, color=:reds )
    p9 = heatmap(v_trained[:,:,2,i], axis=false, cbar=false, aspect_ratio=1, color=:blues )
    e = abs.(u[:,:,2,i] .- u_trained[:,:,2,i])
    p10 = heatmap(e, axis=false, cbar=false, aspect_ratio=1, color=:greens )

    time = round(i*saveat, digits=0)
    fig = plot(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, layout=(2,5), margin =0mm, title="t = $time")

    frame(anim, fig)
end
if isdir("./plots")
    gif(anim, "./plots/02.01-trained_GS.gif", fps=5)
else
    gif(anim, "examples/plots/02.01-trained_GS.gif", fps=5)
end