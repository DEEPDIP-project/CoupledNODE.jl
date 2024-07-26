# # Exploration of SciML (2/2)
# This notebook is the continuation of the previous exploration. Here, we will try to :
# **Use a neural closure term to learn the parameters using a priori fitting.**
# We will explore different implementations of the same problem in the SciML ecosystem:
# a. **NeuraODE + CoupleNODE**: Our implementation to build a coupled problem and solve it using [NeuralODE](https://docs.sciml.ai/DiffEqFlux/stable/layers/NeuralDELayers/#DiffEqFlux.NeuralODE).
# b. **ODEProblem + CoupleNODE**: Same as the previous example but using the emblematic implementation of SciML (ODEProblem) instead of NeuralODE.
# c. **[SplitODE]**(https://docs.sciml.ai/DiffEqDocs/stable/types/split_ode_types/)
# d. 
# We will benchmark some of the key parts and compare the obtained results.

# ## Helper functions
# - Helper function to reshape `ODESolution` to our matrices. Returns an object with dimentions `(x, y, n_samples, t)`
function reshape_ODESolution(ODE_sol, grid)
    u = reshape(ODE_sol[1:(grid.N), :, :], grid.nx, grid.ny, size(ODE_sol, 2), :)
    v = reshape(ODE_sol[(grid_GS.N + 1):end, :, :], grid.nx, grid.ny, size(ODE_sol, 2), :)
    return u, v
end

# - Helper function to plot a Gray-Scott heatmap
function GS_heatmap(data; title = "", color = :reds)
    return heatmap(data,
        axis = false,
        cbar = false,
        aspect_ratio = 1,
        color = color,
        title = title,
        framestyle = :none)
end

# ## Definition of the problem
# As a reminder, the GS model is defined from 
# $\begin{equation}\begin{cases} \frac{du}{dt} = D_u \Delta u - uv^2 + f(1-u)  \equiv F_u(u,v) \\ \frac{dv}{dt} = D_v \Delta v + uv^2 - (f+k)v  \equiv G_v(u,v)\end{cases} \end{equation}$
# In this notebook we are going to try to solve the GS model formulated as follows:
# $\begin{equation}\begin{cases} \frac{du}{dt} = D_u \Delta u - uv^2 + NN_u \\ \frac{dv}{dt} = D_v \Delta v + uv^2 + NN_v \end{cases} \end{equation}$
# where $NN_u$ and $NN_v$ are closure terms that are a neural networks, and account for the parameters $f$ and $(f+k)$ respectively.

# Here we define elements common to all cases: a 2D grid, initial conditions, forces $F_u$ and $G_v$.

# Definition of the grid
import CoupledNODE: Grid, grid_to_linear, linear_to_grid
dx = dy = 1.0
nx = ny = 64
grid_GS = Grid(dim = 2, dx = dx, nx = nx, dy = dy, ny = ny)

# Definition of the initial condition as a random perturbation over a constant background to add variety. 
import Random
rng = Random.seed!(1234);
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
    grid_GS, grid_GS, U₀, V₀, ε_u, ε_v, nsimulations = 20);

# We define the initial condition as a flattened concatenated array
uv0 = vcat(reshape(u_initial, grid_GS.N, :), reshape(v_initial, grid_GS.N, :));

# These are the GS parameters (also used in example 02.01) that we will try to learn
D_u = 0.16
D_v = 0.08
f = 0.055
k = 0.062;

# Exact right hand sides (functions) of the GS model
import CoupledNODE: Laplacian
F_u(u, v) = D_u * Laplacian(u, grid_GS.dx, grid_GS.dy) .- u .* v .^ 2 .+ f .* (1.0 .- u)
G_v(u, v) = D_v * Laplacian(v, grid_GS.dx, grid_GS.dy) .+ u .* v .^ 2 .- (f + k) .* v

# We're going to perform some burnput runs in order to get to an initial state common to all cases
import Lux
import CoupledNODE: create_f_CNODE
# Definition of the CNODE that is not closed (i.e. no neural closure term)
f_burnout = create_f_CNODE((F_u, G_v), (grid_GS, grid_GS); is_closed = false);
θ_0, st_0 = Lux.setup(rng, f_burnout); # placeholder parameters (not meaningful)

import DifferentialEquations: Tsit5
solver_algo = Tsit5(); # defined for all cases
import DiffEqFlux: NeuralODE
trange_burn = (0.0, 1.0)
dt, saveat = (1e-2, 1)
burnout_CNODE = NeuralODE(f_burnout,
    trange_burn,
    solver_algo,
    adaptive = false,
    dt = dt,
    saveat = saveat);
burnout_CNODE_solution = Array(burnout_CNODE(uv0, θ_0, st_0)[1]);

# We will use the last state of the burnout as the initial condition for the next exploration
uv0 = burnout_CNODE_solution[:, :, end];

# We define de time span, dt and saveat cused to generate the solutions that we want to compare.
trange = (0.0, 2000.0);
dt, saveat = (1 / (4 * max(D_u, D_v)), 10);

# Get the **exact solution**: target data that is going to be used to train the neural network.
exact_sol_problem = NeuralODE(
    f_burnout, trange, solver_algo, adaptive = false, dt = dt, saveat = saveat);
exact_sol = Array(exact_sol_problem(uv0, θ_0, st_0)[1]);
# `41.340940 seconds (2.18 M allocations: 298.835 GiB, 31.85% gc time)`
u_exact_sol, v_exact_sol = reshape_ODESolution(exact_sol, grid_GS);
# We create now the data used for training (input) with shape `(nux * nuy + nvx * nvy, nsimulations * timesteps)`
uv_data = reshape(exact_sol, size(exact_sol, 1), size(exact_sol, 2) * size(exact_sol, 3));
# and the labels (forces because we will do apriori fitting)
forces_target = Array(f_burnout(uv_data, θ_0, st_0)[1]);

# Define the forces
import Zygote
function F_u_open(u, v)
    Zygote.@ignore D_u * Laplacian(u, grid_GS.dx, grid_GS.dy) .- u .* v .^ 2
end;
function G_v_open(u, v)
    Zygote.@ignore D_v * Laplacian(v, grid_GS.dx, grid_GS.dy) .+ u .* v .^ 2
end;
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

# ## a. NeuralODE + CoupledNODE
# Create the CNODE with the Neural Network
f_CNODE = create_f_CNODE(
    (F_u_open, G_v_open), (grid_GS, grid_GS), (NN_u, NN_v); is_closed = true);
θ, st = Lux.setup(rng, f_CNODE);

# create loss
import CoupledNODE: create_randloss_derivative
neuralODE_loss = create_randloss_derivative(uv_data, forces_target, f_CNODE, st;
    n_use = 64, λ = 0);

# Set up optimizer
import ComponentArrays
import OptimizationOptimisers: Optimization, OptimiserChain, Adam
adtype = Optimization.AutoZygote();
train_algo = OptimiserChain(Adam(1.0e-3));
opt_f_neuralODE = Optimization.OptimizationFunction((x, p) -> neuralODE_loss(x), adtype);
opt_prob_neuralODE = Optimization.OptimizationProblem(
    opt_f_neuralODE, ComponentArrays.ComponentArray(θ););

# training
import CoupledNODE: callback
result_opt_neuralode = Optimization.solve(opt_prob_neuralODE, train_algo;
    callback = callback, maxiters = 1000);
θ = result_opt_neuralode.u;
opt_prob_neuralODE = Optimization.OptimizationProblem(
    opt_f_neuralODE, ComponentArrays.ComponentArray(θ););

# get solution
neuralODE_problem = NeuralODE(
    f_CNODE, trange, solver_algo, adaptive = false, dt = dt, saveat = saveat);
@time neuralODE_sol = Array(neuralODE_problem(uv0, θ, st)[1]);
# `45.172009 seconds (3.28 M allocations: 317.659 GiB, 14.78% gc time, 0.83% compilation time)`
u_neural_ODE, v_neural_ODE = reshape_ODESolution(neuralODE_sol, grid_GS);

# ### b. ODEProblem
# We first define a set of parameters for training in this setup and a right hand side that is a wrapper 
# written in the way that is valid for defining an `ODEProblem`.
θ_ODE, st_ODE = Lux.setup(rng, f_CNODE);
function rhs_ode_problem(u, p, t)
    f_CNODE(u, θ_ODE, st_ODE)[1]
end

# define the optimization problem and train
opt_prob_ODE = Optimization.OptimizationProblem(
    opt_f_neuralODE, ComponentArrays.ComponentArray(θ_ODE););
result_opt_ode = Optimization.solve(
    opt_prob_ODE, train_algo; callback = callback, maxiters = 1000);
θ_ODE = result_opt_ode.u;
opt_prob_ODE = Optimization.OptimizationProblem(
    opt_f_neuralODE, ComponentArrays.ComponentArray(θ_ODE););

# then create the problem and solve it:
using DifferentialEquations: ODEProblem, solve
ODE_problem = ODEProblem(rhs_ode_problem, uv0, trange);
@time ODE_sol = solve(ODE_problem, solver_algo, adaptive = false, dt = dt, saveat = saveat);
# `45.490858 seconds (3.09 M allocations: 317.156 GiB, 14.74% gc time, 0.75% compilation time)`
u_ODE, v_ODE = reshape_ODESolution(Array(ODE_sol), grid_GS);

# ### c. [SplitODE](https://docs.sciml.ai/DiffEqDocs/stable/types/split_ode_types/)
# `SplitODEProblem` considers a problem with two functions $f_1$ and $f_2$, and it has been used by Hugo Melchers in his Master thesis: [Neural closure models](https://github.com/HugoMelchers/neural-closure-models/blob/main/src/training/models/split_neural_odes.jl).
# We are going to redefine the forces split in two functions:
# $\begin{equation} \frac{du}{dt} f_1(u,p,t) + f_2(u,p,t) \end{equation}$
# We are going to abstract:
# - $f_1$: known part of the equation.
# - $f_2$: Closure term, neural network.
# We define the forces out-of-place because for the loss is easier if we can get the prediction (force).
function f1(u, p, t)
    u_gs = @view u[1, :, :, :]
    v_gs = @view u[2, :, :, :]
    du_gs = F_u_open(u_gs, v_gs)
    dv_gs = G_v_open(u_gs, v_gs)
    return permutedims(cat(du_gs, dv_gs, dims = 4), [4, 1, 2, 3])
end

function f2(u, p, t)
    u_gs = @view u[1, :, :, :]
    v_gs = @view u[2, :, :, :]
    du_gs = NN_u((u_gs, v_gs), p.θ_u, st_u)[1]
    dv_gs = NN_v((u_gs, v_gs), p.θ_v, st_v)[1]
    return permutedims(cat(du_gs, dv_gs, dims = 4), [4, 1, 2, 3])
end

function rhs_split(u, p, t)
    f1(u, p, t) + f2(u, p, t)
end

# Parameters of the neural networks
θ_u, st_u = Lux.setup(rng, NN_u);
θ_v, st_v = Lux.setup(rng, NN_v);
p_split = ComponentArrays.ComponentArray(; θ_u = θ_u, θ_v = θ_v);

# we are going to reshape the inputs and labels to make it consistent with this new formulation
u0 = permutedims(cat(u_initial, v_initial, dims = 4), [4, 1, 2, 3]); # 2x64x64x20 n_vars x nx x ny x n_samples
uv_data_grid = permutedims(cat(u_exact_sol, v_exact_sol, dims = 5), [5, 1, 2, 3, 4]);
uv_data_grid = reshape(uv_data_grid, size(uv_data_grid)[1:3]..., :);
force_u, force_v = reshape_ODESolution(forces_target, grid_GS);
forces_target_grid = permutedims(
    cat(force_u[:, :, :, 1], force_v[:, :, :, 1], dims = 4), [4, 1, 2, 3]);

# NOTE: we cannot use `create_randloss_derivative` because when getting a prediction form the model we fetch the data via `Array(f(u)[1])` but with all ODESolutions we do it as `Array(f(u))`. This is because NeuralODE returns a tuple while other solvers return ODESolutions directly. See [this comment of C. Rackauckas](https://discourse.julialang.org/t/neural-ode-in-diffeqflux-that-is-not-a-time-series/22395/7): "Neural ODE returns the array and not the ODESolution because of the constraints on it’s AD"
function split_loss(θ)
    pred = rhs_split(uv_data_grid, θ, 0)
    return sum(abs2, pred .- forces_target_grid) / sum(abs2, forces_target_grid), nothing
end

opt_f_split = Optimization.OptimizationFunction((x, p) -> split_loss(x), adtype);
opt_prob_split = Optimization.OptimizationProblem(opt_f_split, p_split);

# training
import CoupledNODE: callback
result_opt_split = Optimization.solve(
    opt_prob_split, train_algo; callback = callback, maxiters = 200);
p_split = result_opt_split.u;
opt_prob_split = Optimization.OptimizationProblem(opt_f_split, p_split);
# _Note:_ This model takes very long to train because we are using all the data in the loss, not randomly selected samples (1st hypothesis).

using DifferentialEquations: SplitODEProblem, solve
split_problem = SplitODEProblem(f1, f2, u0, trange);
@time split_sol = solve(split_problem, solver_algo, p = p_split, adaptive = false,
    dt = dt, saveat = saveat, progress = true);
# `57.961440 seconds (5.72 M allocations: 336.088 GiB, 12.60% gc time, 1.39% compilation time)`
u_split = Array(split_sol)[1, :, :, :, :];
v_split = Array(split_sol)[2, :, :, :, :];

# ### d. ODEProblem following [Missing Physics showcase](https://docs.sciml.ai/Overview/stable/showcase/missing_physics/)
#Multilayer FeedForward
rbf(x) = exp.(-(x .^ 2))
const U = Lux.Chain(Lux.Dense(2, 5, rbf), Lux.Dense(5, 5, rbf), Lux.Dense(5, 5, rbf),
    Lux.Dense(5, 2))

#U(u0, p_mp, _st)[1]
#Get the initial parameters and state variables of the model
p_mp, st_mp = Lux.setup(rng, U)
const _st = st_mp

# UDE: universal differential equation : `u' = known(u) + NN(u)` in our case `F_u_open` and `G_v_open` contain part of the known physics. 
# The NNs should approximate the rest.
function ude!(du, u, p, t)
    u_gs = @view u[1, :, :, :]
    v_gs = @view u[2, :, :, :]
    du_gs = @view du[1, :, :, :]
    dv_gs = @view du[2, :, :, :]
    u_nn = U(u, p, _st)[1]
    du_gs .= F_u_open(u_gs, v_gs) + u_nn[1, :, :, :]
    dv_gs .= G_v_open(u_gs, v_gs) + u_nn[2, :, :, :]
end

prob_nn = ODEProblem(ude!, u0, trange, p_mp)
# Trainin loop
using SciMLSensitivity: QuadratureAdjoint, ReverseDiffVJP
function predict(θ, X = u0, T = trange)
    _prob = remake(prob_nn, u0 = X, tspan = (T[1], T[end]), p = θ)
    Array(solve(_prob, solver_algo, saveat = saveat, dt = dt,
        sensealg = QuadratureAdjoint(autojacvec = ReverseDiffVJP(true))))
end

function loss_mp(θ)
    X̂ = predict(θ)
    mean(abs2, forces_target_grid .- X̂)
end
# I think there is a problem here, X̂ is the solution (u,v) not the forces.

# training
optf = Optimization.OptimizationFunction((x, p) -> loss_mp(x), adtype)
optprob = Optimization.OptimizationProblem(
    optf, ComponentArrays.ComponentVector{Float64}(p_mp))
res1 = Optimization.solve(optprob, train_algo, callback = callback, maxiters = 100)

p_trained = res1.u
sol_mp = predict(p_trained, u0, trange)
u_mp = @view sol_mp[1, :, :, :]
v_mp = @view sol_mp[2, :, :, :]

# ### e. NeuralPDE.jl [NNODE](https://docs.sciml.ai/NeuralPDE/stable/manual/ode/#ODE-Specialized-Physics-Informed-Neural-Network-(PINN)-Solver)
# [NNODE](https://docs.sciml.ai/NeuralPDE/stable/tutorials/ode/) is a specific implementation for PINNs such that for an ODE problem:
# \begin{equation} \frac{du}{dt}=f(u,p,t) \end{equation}
# They consider that the solution of the ODE $u \approx NN$ and thus:
# \begin{equation} NN'= f(NN, p, t) \end{equation}
# Let's follow [the tutorial on parameter estimation with PINNs](https://docs.sciml.ai/NeuralPDE/stable/tutorials/ode_parameter_estimation/) 
# to find a NN that the parameters $f$ and $k$ to fit the GS Model.
# We define the force out-of-place because NNODE only supports this type
function GS(u, p, t)
    u_gs = @view u[1, :, :, :]
    v_gs = @view u[2, :, :, :]
    f, k = p #parameters
    du_gs = D_u * Laplacian(u_gs, grid_GS.dx, grid_GS.dy) .- u_gs .* v_gs .^ 2 .+
            f .* (1.0 .- u_gs)
    dv_gs = D_v * Laplacian(v_gs, grid_GS.dx, grid_GS.dy) .+ u_gs .* v_gs .^ 2 .-
            (f + k) .* v_gs
    return permutedims(cat(du_gs, dv_gs, dims = 4), [4, 1, 2, 3])
end

using DifferentialEquations: remake
u0 = permutedims(cat(u_initial, v_initial, dims = 4), [4, 1, 2, 3]); # 2x64x64x20 n_vars x nx x ny x n_samples
prob = ODEProblem(GS, u0, trange, [1.0, 1.0]);
true_p = [f, k]
prob_data = remake(prob, p = true_p);
sol_data = solve(prob_data, solver_algo, saveat = saveat)
#t_ = sol_data.t; # timesteps
u_ = Array(sol_data)
u_ = reshape(u_, size(u_)[1:3]..., :) # merging last two dimensions: n_simualtions and time
labels_ = GS(u_, true_p, true_p)

# We build an MLP with 3 layers with `n` neurons. 
# - input: 1D time (?)
# - output: 2D du, dt (?)
n = 15
NN_PDE = Lux.Chain(
    Lux.Dense(1, n, Lux.σ),
    Lux.Dense(n, n, Lux.σ),
    Lux.Dense(n, n, Lux.σ),
    Lux.Dense(n, 2)
)
ps, st = Lux.setup(rng, NN_PDE) |> Lux.f64
function additional_loss(phi, θ)
    println(size(phi(u_, θ)))
    return sum(abs2, phi(u_, θ) .- labels_) / sum(abs2, labels_)
end

using NeuralPDE: NNODE, WeightedIntervalTraining
strategy_1 = WeightedIntervalTraining([0.7, 0.2, 0.1], 65) # last argument is number of intervals I think it has to match loss dimensions.
# NOTE: tried GridTraining because I saw it in the docs but then found that [is never a good idea to use it](https://github.com/SciML/NeuralPDE.jl/issues/551).
alg_NNODE = NNODE(NN_PDE, train_algo, ps; strategy = strategy_1,
    param_estim = true, additional_loss = additional_loss, dt = dt)
sol = solve(prob, alg_NNODE, verbose = true, abstol = 1e-8,
    maxiters = 5000, dt = dt, saveat = saveat)
#TODO: still not working probably the strategy is not well defined but with the defaults is also a problem. 

# ### f. NeuralPDE.jl and ModelingToolkit.jl
using NeuralPDE, Flux, ModelingToolkit, DiffEqFlux

# Define the parameters for the problem
@parameters t x y
@variables u(..) v(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2

# Define the Gray-Scott equations
eqs = [
    D(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dyy(u(t, x, y)) - u(t, x, y) +
                    u(t, x, y)^2 * v(t, x, y),
    D(v(t, x, y)) ~ Dxx(v(t, x, y)) + Dyy(v(t, x, y)) - u(t, x, y)^2 * v(t, x, y)]

# Define the domains and boundary conditions
domains = [t ∈ IntervalDomain(trange[1], trange[end]),
    x ∈ IntervalDomain(0.0, nx),
    y ∈ IntervalDomain(0.0, ny)]
bcs = [u(0, x, y) ~ cos(pi * x) * cos(pi * y),
    v(0, x, y) ~ sin(pi * x) * sin(pi * y),
    u(t, 0, y) ~ u(t, 1, y),
    u(t, x, 0) ~ u(t, x, 1),
    v(t, 0, y) ~ v(t, 1, y),
    v(t, x, 0) ~ v(t, x, 1)]

# Define the neural networks and the symbolic neural pde

chain = Lux.Chain(Lux.Dense(3, 16, Flux.σ), Lux.Dense(16, 16, Flux.σ), Lux.Dense(16, 2))
nnpde = PhysicsInformedNN(chain, GridTraining(dt), init_params = Flux.glorot_normal)

# Define the discretization
discretization = NeuralPDE.discretize(nnpde, eqs, bcs, domains, dx = [0.1, 0.1, 0.1])

# Define the optimization problem and solve it
prob = GalacticOptim.OptimizationProblem(
    discretization, u0 = [1.0, 1.0], p = nothing, lb = [0.0, 0.0], ub = [1.0, 1.0])
result = GalacticOptim.solve(prob, ADAM(0.1); cb = cb, maxiters = 4000)

# ## Comparison
# we check if there are any differences between the solutions
any(u_ODE - u_neural_ODE .!= 0.0)
any(v_ODE - v_neural_ODE .!= 0.0)
# We see that there are differences between the solutions that can be due to the training.

# ### Plots
using Plots, Plots.PlotMeasures
anim = Animation()
fig = plot(layout = (3, 3), size = (1200, 400))
@gif for i in 1:1:size(u_neural_ODE, 4)
    p1 = GS_heatmap(u_exact_sol[:, :, 1, i], title = "u")
    p2 = GS_heatmap(v_exact_sol[:, :, 1, i], title = "v", color = :blues)
    p3 = GS_heatmap(u_neural_ODE[:, :, 1, i], title = "u")
    p4 = GS_heatmap(v_neural_ODE[:, :, 1, i], title = "v", color = :blues)
    p5 = GS_heatmap(u_split[:, :, 1, i])
    p6 = GS_heatmap(v_split[:, :, 1, i], color = :blues)

    #p5 = GS_heatmap(u_ODE[:, :, 1, i]-u_neural_ODE[:, :, 1, i], color = :greens)
    #p6 = GS_heatmap(v_ODE[:, :, 1, i]-v_ODE[:, :, 1, i], color = :greens)

    #Create titles as separate plots
    t1 = plot(title = "Exact", framestyle = :none)
    t2 = plot(title = "NeuralODE", framestyle = :none)
    t3 = plot(title = "SplitODE", framestyle = :none)
    #t4 = plot(title = "Difference", framestyle = :none)

    fig = plot(t1, p1, p2, t2, p3, p4, t3, p5, p6,
        layout = (3, 3),
        margin = 0mm)
    frame(anim, fig)
end
