using Lux
using SciMLSensitivity
using DiffEqFlux
using DifferentialEquations
using Plots
using Plots.PlotMeasures
using Zygote
using Random
rng = Random.seed!(1234);
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
include("coupling_functions/functions_FDderivatives.jl")
include("coupling_functions/functions_CNODE_loss.jl");

# # Learning the Gray-Scott model: a posteriori fitting

# In this example, we will learn how to approximate a closure to the Gray-Scott model with a Neural Network trained via a posteriori fitting, using a [multishooting](https://docs.sciml.ai/DiffEqFlux/dev/examples/multiple_shooting/) approach.

# As a reminder, the GS model is defined from 
# \begin{equation}\begin{cases} \frac{du}{dt} = D_u \nabla u - uv^2 + f(1-u)  \equiv F_u(u,v) \\ \frac{dv}{dt} = D_v \nabla v + uv^2 - (f+k)v  \equiv G_v(u,v)\end{cases} \end{equation}
# where $u(x,y,t):\mathbb{R}^2\times \mathbb{R}\rightarrow \mathbb{R}$ is the concentration of species 1, while $v(x,y,t)$ is the concentration of species two. This model reproduce the effect of the two species diffusing in their environment, and reacting together.
# This effect is captured by the ratios between $D_u$ and $D_v$ (the diffusion coefficients) and $f$ and $k$ (the reaction rates).


# ## Solving GS to collect data
# Definition of the grid
dux = duy = dvx = dvy = 1.
nux = nuy = nvx = nvy = 64
grid = Grid(dux, duy, nux, nuy, dvx, dvy, nvx, nvy);

# Define the initial condition as a random perturbation over a constant background to add variety. Notice that in this case we are generating only 2 samples (i.e. `nsimulations=2`). This is because for the *a posteriori fitting* we are using a fine sampling.
function initial_condition(grid, U₀, V₀, ε_u, ε_v; nsimulations=1)
    u_init = U₀ .+ ε_u .* randn(grid.nux, grid.nuy, nsimulations)
    v_init = V₀ .+ ε_v .* randn(grid.nvx, grid.nvy, nsimulations)
    return u_init, v_init
end
U₀ = 0.5    # initial concentration of u
V₀ = 0.25   # initial concentration of v
ε_u = 0.05  # magnitude of the perturbation on u
ε_v = 0.1   # magnitude of the perturbation on v
u_initial, v_initial = initial_condition(grid, U₀, V₀, ε_u, ε_v, nsimulations=2);
# $u$ and $v$ are concatenated in a flattended array
uv0 = vcat(reshape(u_initial, grid.Nu,:),reshape(v_initial, grid.Nv,:));

# These are the GS parameters (also used in examples 02.01 and 02.02) that we will try to learn.
D_u = 0.16
D_v = 0.08
f = 0.055
k = 0.062;

# Exact right hand sides of the GS model:
F_u(u, v, grid) = D_u * Laplacian(u, grid.dux, grid.duy) .- u .* v.^2 .+ f .* (1.0 .- u)
G_v(u, v, grid) = D_v * Laplacian(v, grid.dvx, grid.dvy) .+ u .* v.^2 .- (f + k) .* v

# CNODE definition
f_CNODE = create_f_CNODE(F_u, G_v, grid; is_closed=false);
θ_0, st_0 = Lux.setup(rng, f_CNODE);

# **Burnout run**
trange_burn = (0.0f0, 1.0f0)
dt, saveat = (1e-2, 1)
burnout_CNODE = NeuralODE(f_CNODE, trange_burn, Tsit5(), adaptive=false, dt=dt, saveat=saveat);
burnout_CNODE_solution = Array(burnout_CNODE(uv0, θ_0, st_0)[1]);
# Second burnout with a larger timestep
trange_burn = (0.0f0, 800.0f0)
dt, saveat = (1/(4 * max(D_u, D_v)), 100)
burnout_CNODE = NeuralODE(f_CNODE, trange_burn, Tsit5(), adaptive=false, dt=dt, saveat=saveat);
burnout_CNODE_solution = Array(burnout_CNODE(burnout_CNODE_solution[:,:,end], θ_0, st_0)[1]);

# Data collection run
uv0 = burnout_CNODE_solution[:,:,end];
# For the a-posteriori fitting, we use a fine sampling for two reasons:
# 1. use a handlable amount of data points
# 2. prevent instabilities while training
# However, this means that the simulation can not be long.
dt, saveat = (1/(4*max(D_u,D_v)), 0.001)
trange = (0.0f0, 50.0f0)
GS_CNODE = NeuralODE(f_CNODE, trange, Tsit5(), adaptive=false, dt=dt, saveat=saveat);
GS_sim = Array(GS_CNODE(uv0, θ_0, st_0)[1])


# ## Training a CNODE to learn the GS model via a posteriori training
# To learn the GS model, we will use the following CNODE
# \begin{equation}\begin{cases} \frac{du}{dt} = D_u \nabla u + \theta_{u,1} uv^2 +\theta_{u,2} v^2u + \theta_{u,3} u +\theta_{u,4} v +\theta_{u,5}  \\ \frac{dv}{dt} = D_v \nabla v + \theta_{v,1} uv^2 + \theta_{v,2} v^2u +\theta_{v,3} u +\theta_{v,4} v +\theta_{v,5} \end{cases} \end{equation}
# In this example the deterministic function contains the diffusion and the coupling terms, while the model has to learn the source and death terms.
# Then the deterministic functions of the two coupled equations are
F_u_open(u, v, grid) = Zygote.@ignore D_u * Laplacian(u, grid.dux, grid.duy) .- u .* v.^2
G_v_open(u, v, grid) = Zygote.@ignore D_v * Laplacian(v, grid.dvx, grid.dvy) .+ u .* v.^2 
# We tell Zygote to ignore this tree branch for the gradient propagation.


# ### Definition of Neural functions
# In this case we define different architectures for $u$: $NN_u$ and $v$: $NN_v$, to make the training easier.
struct GSLayer_u{F} <: Lux.AbstractExplicitLayer
    init_weight::F
end
function GSLayer_u(; init_weight = zeros32)
    return GSLayer_u(init_weight)
end
struct GSLayer_v{F} <: Lux.AbstractExplicitLayer
    init_weight::F
end
function GSLayer_v(; init_weight = zeros32)
    return GSLayer_v(init_weight)
end

Lux.initialparameters(rng::AbstractRNG, (; init_weight)::GSLayer_u) = (;
    gs_weights = init_weight(rng, 2),
)
Lux.initialparameters(rng::AbstractRNG, (; init_weight)::GSLayer_v) = (;
    gs_weights = init_weight(rng, 1),
)
Lux.initialstates(::AbstractRNG, ::GSLayer_u) = (;)
Lux.initialstates(::AbstractRNG, ::GSLayer_v) = (;)
Lux.parameterlength((; )::GSLayer_u) = 2
Lux.parameterlength((; )::GSLayer_v) = 1
Lux.statelength(::GSLayer_u) = 0
Lux.statelength(::GSLayer_v) = 0
function ((;)::GSLayer_u)(x, params, state)
    u = x[:,:,1,:]
    out = params.gs_weights[1] .* u .+ params.gs_weights[2] 
    out, state
end
function ((;)::GSLayer_v)(x, params, state)
    v = x[:,:,2,:]
    out = params.gs_weights[1] .* v
    out, state
end

# The trainable parts are thus:
NN_u = GSLayer_u()
NN_v = GSLayer_v()

# Close the CNODE with the Neural Network
f_closed_CNODE = create_f_CNODE(F_u_open, G_v_open, grid, NN_u, NN_v; is_closed=true) 
θ, st = Lux.setup(rng, f_closed_CNODE);
θ = ComponentArray(θ)


# ### Design the loss function - a posteriori fitting
# In *a posteriori* fitting, we rely on a differentiable solver to propagate the gradient through the solution of the NODE. In this case, we use the *multishooting a posteriori* fitting [(MulDtO)](https://docs.sciml.ai/DiffEqFlux/dev/examples/multiple_shooting/), where we use `Zygote` to compare `nintervals` of length `nunroll` to get the gradient. Notice that this method is differentiating through the solution of the NODE!
nunroll = 10   
nintervals = 5
nsamples = 1 ;
# Since we want to control the time step and the total length of the solutions that we have to compute at each iteration, we  define an auxiliary NODE that will be used for training. 
# In particular, we can use smaller time steps for the training, but have to sample at the same rate as the data.
# Also, it is important to solve for only the time interval thas is needed at each training step (corresponding to `nunroll` steps)
dt_train = 0.001;
saveat_train = saveat
t_train_range = (0.0f0, saveat_train * nunroll)
training_CNODE = NeuralODE(f_closed_CNODE, t_train_range, Tsit5(), adaptive=false, dt=dt_train, saveat=saveat_train);
# Let's also define a secondary auxiliary CNODE that will be used in case the previous one is unstable
t_train_range_2 = (0.0f0, saveat_train*2)
t_train_range_2 = t_train_range
training_CNODE_2 = NeuralODE(f_closed_CNODE, t_train_range_2, Tsit5(), adaptive=false, dt=0.5 * dt_train, saveat=saveat_train);

# Create the loss
myloss = create_randloss_MulDtO(GS_sim, nunroll=nunroll, nintervals=nintervals, nsamples=nsamples, λ_c=1e2, λ_l1=1e-1);

# To initialize the training, we need some objects to monitor the procedure, and we trigger the first compilation.
lhist = Float32[];
# Initialize and trigger the compilation of the model
pinit = ComponentArray(θ)
myloss(pinit)  # trigger compilation
# ⚠️ Check that the loss does not get type warnings, otherwise it will be slower


# ### Autodifferentiation type
adtype = Optimization.AutoZygote();
# We transform the NeuralODE into an optimization problem
optf = Optimization.OptimizationFunction((x, p) -> myloss(x), adtype);
optprob = Optimization.OptimizationProblem(optf, pinit);

# ### Training algorithm
# In this example we use a quasi-newtonian method
using OptimizationOptimJL
using LineSearches
algo = LBFGS(linesearch = LineSearches.BackTracking(order=3));

# ### Train the CNODEs
result_neuralode = Optimization.solve(optprob,
    algo;
    callback = callback,
    maxiters = 50,
    );
pinit = result_neuralode.u;
θ = pinit
optprob = Optimization.OptimizationProblem(optf, pinit);

# ## Analyse the results

# ### Comparison: learned weights vs (expected) values of the parameters
correct_w_u = [-f, f ]
correct_w_v = [-(f + k)]
gs_w_u = θ.layer_3.layer_1.gs_weights
gs_w_v = θ.layer_3.layer_2.gs_weights
p1 = scatter(gs_w_u, label="learned", title="Comparison NN_u coefficients", xlabel="Index", ylabel="Value")
scatter!(p1, correct_w_u, label="correct")
p2 = scatter(gs_w_v, label="learned", title="Comparison NN_v coefficients", xlabel="Index", ylabel="Value")
scatter!(p2, correct_w_v, label="correct")
p = plot(p1, p2, layout = (2, 1))
display(p)
# The learned weights look perfect, let's check what happens if we use them to solve the GS model.

# ### Comparison: CNODE vs exact solutions

trange = (0.0f0, 600)
dt, saveat = (1, 5)

# Exact solution
f_exact = create_f_CNODE(F_u, G_v, grid; is_closed=false) 
θ_e, st_e = Lux.setup(rng, f_exact);
exact_CNODE = NeuralODE(f_exact, trange, Tsit5(), adaptive=false, dt=dt, saveat=saveat);
exact_CNODE_solution = Array(exact_CNODE(GS_sim[:,1:2,1], θ_e, st_e)[1]);
u = reshape(exact_CNODE_solution[1:grid.Nu, :, :]    , grid.nux, grid.nuy, size(exact_CNODE_solution,2), :);
v = reshape(exact_CNODE_solution[grid.Nu+1:end, :, :], grid.nvx, grid.nvy, size(exact_CNODE_solution,2), :);

# Trained solution
trained_CNODE = NeuralODE(f_closed_CNODE, trange, Tsit5(), adaptive=false, dt=dt, saveat=saveat);
trained_CNODE_solution = Array(trained_CNODE(GS_sim[:,1:2,1], θ, st)[1]);
u_trained = reshape(trained_CNODE_solution[1:grid.Nu, :, :]    , grid.nux, grid.nuy, size(trained_CNODE_solution,2), :);
v_trained = reshape(trained_CNODE_solution[grid.Nu+1:end, :, :], grid.nvx, grid.nvy, size(trained_CNODE_solution,2), :);

# Untrained solution
f_u = create_f_CNODE(F_u_open, G_v_open, grid, NN_u, NN_v; is_closed=false) 
θ_u, st_u = Lux.setup(rng, f_u);
untrained_CNODE = NeuralODE(f_u, trange, Tsit5(), adaptive=false, dt=dt, saveat=saveat);
untrained_CNODE_solution = Array(untrained_CNODE(GS_sim[:,1:2,1], θ_u, st_u)[1]);
u_untrained = reshape(untrained_CNODE_solution[1:grid.Nu, :, :]    , grid.nux, grid.nuy, size(untrained_CNODE_solution,2), :);
v_untrained = reshape(untrained_CNODE_solution[grid.Nu+1:end, :, :], grid.nvx, grid.nvy, size(untrained_CNODE_solution,2), :);

# ### Plot the results
anim = Animation()
fig = plot(layout = (2, 6), size = (1200, 400))
@gif for i in 1:1:size(u_trained, 4)
    ## First row: set of parameters 1
    p1 = heatmap(u[:,:,1,i], axis=false, cbar=false, aspect_ratio=1, color=:reds, title="Exact" )
    p2 = heatmap(v[:,:,1,i], axis=false, cbar=false, aspect_ratio=1, color=:blues)
    p3 = heatmap(u_untrained[:,:,1,i], axis=false, cbar=false, aspect_ratio=1, color=:reds, title="Untrained")
    p4 = heatmap(v_untrained[:,:,1,i], axis=false, cbar=false, aspect_ratio=1, color=:blues)
    p5 = heatmap(u_trained[:,:,1,i], axis=false, cbar=false, aspect_ratio=1, color=:reds, title="Trained")
    p6 = heatmap(v_trained[:,:,1,i], axis=false, cbar=false, aspect_ratio=1, color=:blues)

    ## Second row: set of parameters 2
    p7 = heatmap(u[:,:,2,i], axis=false, cbar=false, aspect_ratio=1, color=:reds, title="Exact" )
    p8 = heatmap(v[:,:,2,i], axis=false, cbar=false, aspect_ratio=1, color=:blues)
    p9 = heatmap(u_untrained[:,:,2,i], axis=false, cbar=false, aspect_ratio=1, color=:reds, title="Untrained")
    p10 = heatmap(v_untrained[:,:,2,i], axis=false, cbar=false, aspect_ratio=1, color=:blues)
    p11 = heatmap(u_trained[:,:,2,i], axis=false, cbar=false, aspect_ratio=1, color=:reds, title="Trained")
    p12 = heatmap(v_trained[:,:,2,i], axis=false, cbar=false, aspect_ratio=1, color=:blues)

    fig = plot(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, layout=(2,6), margin=0mm)
    frame(anim, fig)
end

# Save the generated .gif
if isdir("./plots")
    gif(anim, "./plots/02.02-trained_GS.gif", fps=10)
else
    gif(anim, "examples/plots/02.02-trained_GS.gif", fps=10)
end
