# # Learning the Gray-Scott model: a priori fitting
# In the previous example [02.00-GrayScott.jl](02.00-GrayScott.jl) we have seen how to use the CNODE to solve the Gray-Scott model via an explicit method.
# Here we introduce the *Learning part* of the CNODE, and show how it can be used to close the CNODE. We are going to train the neural network via **a priori fitting** and in the next example [02.02-GrayScott](02.02-GrayScott.jl) we will show how to use a posteriori fitting.

# As a reminder, the GS model is defined from 
# $\begin{equation}\begin{cases} \frac{du}{dt} = D_u \Delta u - uv^2 + f(1-u)  \equiv F_u(u,v) \\ \frac{dv}{dt} = D_v \Delta v + uv^2 - (f+k)v  \equiv G_v(u,v)\end{cases} \end{equation}$
# where $u(x,y,t):\mathbb{R}^2\times \mathbb{R}\rightarrow \mathbb{R}$ is the concentration of species 1, while $v(x,y,t)$ is the concentration of species two. This model reproduce the effect of the two species diffusing in their environment, and reacting together.
# This effect is captured by the ratios between $D_u$ and $D_v$ (diffusion coefficients) and $f$ and $k$ (reaction rates).

# In this example, we will first (I) use the exact GS model to gather some data
# then in the second part (II), we will train a generic CNODE to show that it can learn the GS model from the data.

# ## I. Solving GS to collect data
# Definition of the grid
import CoupledNODE: Grid
dux = duy = dvx = dvy = 1.0
nux = nuy = nvx = nvy = 64
grid_GS_u = Grid(dim = 2, dx = dux, nx = nux, dy = duy, ny = nuy)
grid_GS_v = Grid(dim = 2, dx = dvx, nx = nvx, dy = dvy, ny = nvy)

# Definition of the initial condition as a random perturbation over a constant background to add variety. 
# Notice that this initial conditions are different from those of the previous example.
import Random
function initial_condition(grid_u, grid_v, U₀, V₀, ε_u, ε_v; nsimulations = 1)
    u_init = U₀ .+ ε_u .* Random.randn(grid_u.nx, grid_u.ny, nsimulations)
    v_init = V₀ .+ ε_v .* Random.randn(grid_v.nx, grid_v.ny, nsimulations)
    return u_init, v_init
end
U₀ = 0.5    # initial concentration of u
V₀ = 0.25   # initial concentration of v
ε_u = 0.05  # magnitude of the perturbation on u
ε_v = 0.1   # magnitude of the perturbation on v
u_initial, v_initial = initial_condition(
    grid_GS_u, grid_GS_v, U₀, V₀, ε_u, ε_v, nsimulations = 20);

# We define the initial condition as a flattened concatenated array
uv0 = vcat(reshape(u_initial, grid_GS_u.N, :), reshape(v_initial, grid_GS_v.N, :));

# These are the GS parameters (also used in example 02.01) that we will try to learn
D_u = 0.16
D_v = 0.08
f = 0.055
k = 0.062;

# Exact right hand sides (functions) of the GS model
import CoupledNODE: Laplacian
F_u(u, v) = D_u * Laplacian(u, grid_GS_u.dx, grid_GS_u.dy) .- u .* v .^ 2 .+ f .* (1.0 .- u)
G_v(u, v) = D_v * Laplacian(v, grid_GS_v.dx, grid_GS_v.dy) .+ u .* v .^ 2 .- (f + k) .* v

# Definition of the CNODE
import Lux
import CoupledNODE: create_f_CNODE
f_CNODE = create_f_CNODE((F_u, G_v), (grid_GS_u, grid_GS_v); is_closed = false);
rng = Random.seed!(1234);
θ_0, st_0 = Lux.setup(rng, f_CNODE);

# **Burnout run:** to discard the results of the initial conditions.
# In this case we need 2 burnouts: first one with a relatively large time step and then another one with a smaller time step. This allow us to discard the transient dynamics and to have a good initial condition for the data collection run.
import DifferentialEquations: Tsit5
import DiffEqFlux: NeuralODE
trange_burn = (0.0, 1.0)
dt, saveat = (1e-2, 1)
burnout_CNODE = NeuralODE(f_CNODE,
    trange_burn,
    Tsit5(),
    adaptive = false,
    dt = dt,
    saveat = saveat);
burnout_CNODE_solution = Array(burnout_CNODE(uv0, θ_0, st_0)[1]);
# Second burnout with a smaller timestep
trange_burn = (0.0, 500.0)
dt, saveat = (1 / (4 * max(D_u, D_v)), 100)
burnout_CNODE = NeuralODE(f_CNODE,
    trange_burn,
    Tsit5(),
    adaptive = false,
    dt = dt,
    saveat = saveat);
burnout_CNODE_solution = Array(burnout_CNODE(burnout_CNODE_solution[:, :, end], θ_0, st_0)[1]);

# Data collection run
uv0 = burnout_CNODE_solution[:, :, end];
trange = (0.0, 2000.0);
dt, saveat = (1 / (4 * max(D_u, D_v)), 1);
GS_CNODE = NeuralODE(f_CNODE, trange, Tsit5(), adaptive = false, dt = dt, saveat = saveat);
GS_sim = Array(GS_CNODE(uv0, θ_0, st_0)[1]);
# `GS_sim` contains the solutions of $u$ and $v$ for the specified `trange` and `nsimulations` initial conditions. If you explore `GS_sim` you will see that it has the shape `((nux * nuy) + (nvx * nvy), nsimulations, timesteps)`.
# `uv_data` is a reshaped version of `GS_sim` that has the shape `(nux * nuy + nvx * nvy, nsimulations * timesteps)`. This is the format that we will use to train the CNODE.
uv_data = reshape(GS_sim, size(GS_sim, 1), size(GS_sim, 2) * size(GS_sim, 3));
# We define `FG_target` containing the right hand sides (i.e. $\frac{du}{dt} and \frac{dv}{dt}$) of each one of the samples. We will see later that for the training `FG_target` is used as the labels to do derivative fitting.
FG_target = Array(f_CNODE(uv_data, θ_0, st_0)[1]);

# ## II. Training a CNODE to learn the GS model via a priori training
# To learn the GS model, we will use the following CNODE
# \begin{equation}\begin{cases} \frac{du}{dt} = D_u \Delta u - uv^2 + \theta_{u,1} u +\theta_{u,2} v +\theta_{u,3}  \\ \frac{dv}{dt} = D_v \Delta v + uv^2 + \theta_{v,1} u +\theta_{v,2} v +\theta_{v,3} \end{cases} \end{equation}
# In this example the deterministic function contains the diffusion and the coupling terms, while the model has to learn the source and death terms.
# The deterministic functions of the two coupled equations are:
import Zygote
function F_u_open(u, v)
    Zygote.@ignore D_u * Laplacian(u, grid_GS_u.dx, grid_GS_u.dy) .- u .* v .^ 2
end;
function G_v_open(u, v)
    Zygote.@ignore D_v * Laplacian(v, grid_GS_v.dx, grid_GS_v.dy) .+ u .* v .^ 2
end;
# We tell Zygote to ignore this tree branch for the gradient propagation.

# For the trainable part, we define an abstract Lux layer
struct GSLayer{F} <: Lux.AbstractExplicitLayer
    init_weight::F
end
# and its (outside) constructor
function GSLayer(; init_weight = Lux.zeros32)
    #function GSLayer(; init_weight = Lux.glorot_uniform)
    return GSLayer(init_weight)
end
# We also need to specify how to initialize its parameters and states. 
# This custom layer does not have any hidden states (RNGs) that are modified.
function Lux.initialparameters(rng::Random.AbstractRNG, (; init_weight)::GSLayer)
    (;
        gs_weights = init_weight(rng, 3),)
end
Lux.initialstates(::Random.AbstractRNG, ::GSLayer) = (;)
Lux.parameterlength((;)::GSLayer) = 3
Lux.statelength(::GSLayer) = 0

# We define how to pass inputs through `GSlayer`, assuming the following:
# - Input size: `(N, N, 2, nsample)`, where the two channels are $u$ and $v$.
# - Output size: `(N, N, nsample)` where we assumed monochannel output, so we dropped the channel dimension.

# This is what each layer does. Notice that the layer does not modify the state.
function ((;)::GSLayer)(x, params, state)
    (u, v) = x
    out = params.gs_weights[1] .* u .+ params.gs_weights[2] .* v .+ params.gs_weights[3]
    out, state
end
# We create the trainable models. In this case is just a GS layer, but the model can be as complex as needed.
NN_u = GSLayer()
NN_v = GSLayer()

# We can now close the CNODE with the Neural Network
f_closed_CNODE = create_f_CNODE(
    (F_u_open, G_v_open), (grid_GS_u, grid_GS_v), (NN_u, NN_v); is_closed = true)
θ, st = Lux.setup(rng, f_closed_CNODE);
print(θ)

# Check that the closed CNODE can reproduce the GS model if the parameters are set to the correct values 
import ComponentArrays
correct_w_u = [-f, 0, f];
correct_w_v = [0, -(f + k), 0];
θ_correct = ComponentArrays.ComponentArray(θ)
θ_correct.layer_2.layer_1.gs_weights = correct_w_u;
θ_correct.layer_2.layer_2.gs_weights = correct_w_v;

# Notice that they are the same within a tolerance of 1e-7
isapprox(f_closed_CNODE(GS_sim[:, 1, 1], θ_correct, st)[1],
    f_CNODE(GS_sim[:, 1, 1], θ_0, st_0)[1],
    atol = 1e-7,
    rtol = 1e-7)
# but now with a tolerance of 1e-8 this check returns `false`.
isapprox(f_closed_CNODE(GS_sim[:, 1, 1], θ_correct, st)[1],
    f_CNODE(GS_sim[:, 1, 1], θ_0, st_0)[1],
    atol = 1e-8,
    rtol = 1e-8)
# In a chaotic system like GS, this would be enough to produce different dynamics, so be careful about this

# If you have problems with training the model, you can cheat and start from the solution to check your implementation:
# ```julia
# θ.layer_2.layer_1.gs_weights = correct_w_u
# θ.layer_2.layer_2.gs_weights = correct_w_v
# pinit = θ
# ```

# ### Design the loss function - a priori fitting
# For this example, we use *a priori* fitting. In this approach, the loss function is defined to minimize the difference between the derivatives of $\frac{du}{dt}$ and $\frac{dv}{dt}$ predicted by the model and calculated via explicit method `FG_target`.
# In practice, we use [Zygote](https://fluxml.ai/Zygote.jl/stable/) to compare the right hand side of the GS model with the right hand side of the CNODE, and we ask it to minimize the difference.
import CoupledNODE: create_randloss_derivative
myloss = create_randloss_derivative(uv_data,
    FG_target,
    f_closed_CNODE,
    st;
    n_use = 64,
    λ = 0);

## Initialize and trigger the compilation of the model
pinit = ComponentArrays.ComponentArray(θ);
myloss(pinit);
## [!] Check that the loss does not get type warnings, otherwise it will be slower

# We transform the NeuralODE into an optimization problem
## Select the autodifferentiation type
import OptimizationOptimisers: Optimization
adtype = Optimization.AutoZygote();
optf = Optimization.OptimizationFunction((x, p) -> myloss(x), adtype);
optprob = Optimization.OptimizationProblem(optf, pinit);

# Select the training algorithm:
# In the previous example we have used a classic gradient method like Adam:
import OptimizationOptimisers: OptimiserChain, Adam
algo = OptimiserChain(Adam(1.0e-3));
# notice however that CNODEs can be trained with any Julia optimizer, including the ones from the `Optimization` package like LBFGS
import OptimizationOptimJL: Optim
algo = Optim.LBFGS();
# or even gradient-free methods like CMA-ES that we use for this example
using OptimizationCMAEvolutionStrategy, Statistics
algo = CMAEvolutionStrategyOpt();

# ### Train the CNODE
import CoupledNODE: callback
result_neuralode = Optimization.solve(optprob,
    algo;
    callback = callback,
    maxiters = 150);
pinit = result_neuralode.u;
θ = pinit;
optprob = Optimization.OptimizationProblem(optf, pinit);
# (Notice that the block above can be repeated to continue training, however don't do that with CMA-ES since it will restart from a random initial population)

# ## III. Analyse the results
# Let's compare the learned weights to the values that we expect
using Plots, Plots.PlotMeasures
gs_w_u = θ.layer_2.layer_1.gs_weights;
gs_w_v = θ.layer_2.layer_2.gs_weights;
p1 = scatter(gs_w_u,
    label = "learned",
    title = "Comparison NN_u coefficients",
    xlabel = "Index",
    ylabel = "Value")
scatter!(p1, correct_w_u, label = "correct")
p2 = scatter(gs_w_v,
    label = "learned",
    title = "Comparison NN_v coefficients",
    xlabel = "Index",
    ylabel = "Value")
scatter!(p2, correct_w_v, label = "correct")
p = plot(p1, p2, layout = (2, 1))
display(p)
# The learned weights look perfect, but let's check what happens if we use them to solve the GS model.

# Let's solve the system, for two different set of parameters, with the trained CNODE and compare with the exact solution
trange = (0.0, 500);
dt, saveat = (1, 5);

## Exact solution
f_exact = create_f_CNODE((F_u, G_v), (grid_GS_u, grid_GS_v); is_closed = false)
θ_e, st_e = Lux.setup(rng, f_exact);
exact_CNODE = NeuralODE(f_exact,
    trange,
    Tsit5(),
    adaptive = false,
    dt = dt,
    saveat = saveat);
exact_CNODE_solution = Array(exact_CNODE(GS_sim[:, 1:2, 1], θ_e, st_e)[1]);
u = reshape(exact_CNODE_solution[1:(grid_GS_u.N), :, :],
    grid_GS_u.nx,
    grid_GS_u.ny,
    size(exact_CNODE_solution, 2),
    :);
v = reshape(exact_CNODE_solution[(grid_GS_v.N + 1):end, :, :],
    grid_GS_v.nx,
    grid_GS_v.ny,
    size(exact_CNODE_solution, 2),
    :);

## Trained solution
trained_CNODE = NeuralODE(f_closed_CNODE,
    trange,
    Tsit5(),
    adaptive = false,
    dt = dt,
    saveat = saveat);
trained_CNODE_solution = Array(trained_CNODE(GS_sim[:, 1:3, 1], θ, st)[1]);
u_trained = reshape(trained_CNODE_solution[1:(grid_GS_u.N), :, :],
    grid_GS_u.nx,
    grid_GS_u.ny,
    size(trained_CNODE_solution, 2),
    :);
v_trained = reshape(trained_CNODE_solution[(grid_GS_u.N + 1):end, :, :],
    grid_GS_v.nx,
    grid_GS_v.ny,
    size(trained_CNODE_solution, 2),
    :);

f_u = create_f_CNODE(
    (F_u_open, G_v_open), (grid_GS_u, grid_GS_v), (NN_u, NN_v); is_closed = true)
# Setting these new parameters ensure us that we do not use the trained network.
θ_u, st_u = Lux.setup(rng, f_u);

## Untrained solution
untrained_CNODE = NeuralODE(f_u,
    trange,
    Tsit5(),
    adaptive = false,
    dt = dt,
    saveat = saveat);
untrained_CNODE_solution = Array(untrained_CNODE(GS_sim[:, 1:3, 1], θ_u, st_u)[1]);
u_untrained = reshape(untrained_CNODE_solution[1:(grid_GS_u.N), :, :],
    grid_GS_u.nx,
    grid_GS_u.ny,
    size(untrained_CNODE_solution, 2),
    :);
v_untrained = reshape(untrained_CNODE_solution[(grid_GS_v.N + 1):end, :, :],
    grid_GS_v.nx,
    grid_GS_v.ny,
    size(untrained_CNODE_solution, 2),
    :);

# ### Plot the results
anim = Animation()
fig = plot(layout = (2, 6), size = (1200, 400))
@gif for i in 1:1:size(u_trained, 4)
    ## First row: set of parameters 1
    p1 = heatmap(u[:, :, 1, i],
        axis = false,
        cbar = false,
        aspect_ratio = 1,
        color = :reds,
        title = "Exact")
    p2 = heatmap(v[:, :, 1, i],
        axis = false,
        cbar = false,
        aspect_ratio = 1,
        color = :blues)
    p3 = heatmap(u_untrained[:, :, 1, i],
        axis = false,
        cbar = false,
        aspect_ratio = 1,
        color = :reds,
        title = "Untrained")
    p4 = heatmap(v_untrained[:, :, 1, i],
        axis = false,
        cbar = false,
        aspect_ratio = 1,
        color = :blues)
    p5 = heatmap(u_trained[:, :, 1, i],
        axis = false,
        cbar = false,
        aspect_ratio = 1,
        color = :reds,
        title = "Trained")
    p6 = heatmap(v_trained[:, :, 1, i],
        axis = false,
        cbar = false,
        aspect_ratio = 1,
        color = :blues)

    ## Second row: set of parameters 2
    p7 = heatmap(u[:, :, 2, i],
        axis = false,
        cbar = false,
        aspect_ratio = 1,
        color = :reds,
        title = "Exact")
    p8 = heatmap(v[:, :, 2, i],
        axis = false,
        cbar = false,
        aspect_ratio = 1,
        color = :blues)
    p9 = heatmap(u_untrained[:, :, 2, i],
        axis = false,
        cbar = false,
        aspect_ratio = 1,
        color = :reds,
        title = "Untrained")
    p10 = heatmap(v_untrained[:, :, 2, i],
        axis = false,
        cbar = false,
        aspect_ratio = 1,
        color = :blues)
    p11 = heatmap(u_trained[:, :, 2, i],
        axis = false,
        cbar = false,
        aspect_ratio = 1,
        color = :reds,
        title = "Trained")
    p12 = heatmap(v_trained[:, :, 2, i],
        axis = false,
        cbar = false,
        aspect_ratio = 1,
        color = :blues)

    fig = plot(p1,
        p2,
        p3,
        p4,
        p5,
        p6,
        p7,
        p8,
        p9,
        p10,
        p11,
        p12,
        layout = (2, 6),
        margin = 0mm)
    frame(anim, fig)
end

# Save the generated .gif
if isdir("./plots")
    gif(anim, "./plots/02.01-trained_GS.gif", fps = 10)
else
    gif(anim, "examples/plots/02.01-trained_GS.gif", fps = 10)
end
# Notice that even with a posteriori loss of the order of 1e-7 still produces a different dynamics over time!
myloss(θ)
