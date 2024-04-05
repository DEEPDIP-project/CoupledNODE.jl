import CUDA
import DifferentialEquations: Tsit5
import DiffEqGPU: GPUTsit5
const solver_algo = CUDA.functional() ? GPUTsit5() : Tsit5();
const MY_TYPE = Float32 # use float32 if you plan to use a GPU
if CUDA.functional()
    CUDA.allowscalar(false)
end

# # Learning a closure for the Gray-Scott model when using a coarser grid for $v$
# In this example, we want to learn a closure to the Gray-Scott model using a Neural Network. Here we use a **coarser grid only on $v$**, while we keep the **fine grid for $u$**.
# ## Exact solution in *fine* grid - DNS
# We run multiple GS simulations as discussed in the previous part.
# *Note:* the *fine* grid is only 40 cells per side to speed up the example.
dux = duy = dvx = dvy = 1.0f0
nux = nuy = nvx = nvy = 40
import CoupledNODE: Grid, Laplacian
grid_GS = Grid(dux, duy, nux, nuy, dvx, dvy, nvx, nvy, convert_to_float32 = true);
# Let's define the initial condition as a random perturbation over a constant background to add variety.
function initial_condition(grid, U₀, V₀, ε_u, ε_v; nsimulations = 1)
    u_init = U₀ .+ ε_u .* randn(grid.nux, grid.nuy, nsimulations)
    v_init = V₀ .+ ε_v .* randn(grid.nvx, grid.nvy, nsimulations)
    return u_init, v_init
end
U₀ = 0.5f0    # initial concentration of u
V₀ = 0.25f0   # initial concentration of v
ε_u = 0.05f0  # magnitude of the perturbation on u
ε_v = 0.1f0   # magnitude of the perturbation on v
u_initial, v_initial = initial_condition(grid_GS, U₀, V₀, ε_u, ε_v, nsimulations = 4);

# We can now define the initial condition as a flattened concatenated array.
uv0 = MY_TYPE.(vcat(
    reshape(u_initial, grid_GS.Nu, :), reshape(v_initial, grid_GS.nvx * grid_GS.nvy, :)));

# From the literature, we have selected the following parameters in order to form nice patterns
D_u = 0.16f0
D_v = 0.08f0
f = 0.055f0
k = 0.062f0;

# Definiton of the right-hand-sides (RHS) of GS model
import Zygote
function F_u(u, v, grid)
    Zygote.@ignore MY_TYPE.(D_u * Laplacian(u, grid.dux, grid.duy) .- u .* v .^ 2 .+
                            f .* (1.0f0 .- u))
end
function G_v(u, v, grid)
    Zygote.@ignore MY_TYPE.(D_v * Laplacian(v, grid.dvx, grid.dvy) .+ u .* v .^ 2 .-
                            (f + k) .* v)
end
# Definition of the model
import CoupledNODE: create_f_CNODE
f_CNODE = create_f_CNODE(F_u, G_v, grid_GS; is_closed = false);
import Random, LuxCUDA, Lux
rng = Random.seed!(1234)
θ, st = Lux.setup(rng, f_CNODE);

# Short *burnout run* to get rid of the initial artifacts
trange_burn = (0.0f0, 50.0f0)
dt, saveat = (1e-2, 1)
# [!] According to [DiffEqGPU docs](https://docs.sciml.ai/DiffEqGPU/stable/getting_started/) 
#     the best thing to do in case of bottleneck consisting in expensive right hand side is to 
#     use `CuArray` as initial condition and do not rely on `EnsemblesGPU`.
using DiffEqFlux: NeuralODE
full_CNODE = NeuralODE(f_CNODE,
    trange_burn,
    solver_algo,
    adaptive = false,
    dt = dt,
    saveat = saveat);
burnout_data = Array(full_CNODE(uv0, θ, st)[1]);

# Second burnout with larger timesteps
trange_burn = (0.0f0, 100.0f0)
dt, saveat = (1, 50)
full_CNODE = NeuralODE(f_CNODE,
    trange_burn,
    solver_algo,
    adaptive = false,
    dt = dt,
    saveat = saveat);
burnout_data = Array(full_CNODE(burnout_data[:, :, end], θ, st)[1]);
u = reshape(
    burnout_data[1:(grid_GS.Nu), :, :], grid_GS.nux, grid_GS.nuy, size(burnout_data, 2), :);
v = reshape(burnout_data[(grid_GS.Nu + 1):end, :, :],
    grid_GS.nvx,
    grid_GS.nvy,
    size(burnout_data, 2),
    :);

# ### Data collection run - exact solution in *fine* grid
# We use the output of the burnout to start a longer simulation.
uv0 = burnout_data[:, :, end];
trange = (0.0f0, 2500.0f0)
# for this data production run, we set `dt=1` and we sample every step.
dt, saveat = (0.1, 0.1)
full_CNODE = NeuralODE(
    f_CNODE, trange, solver_algo, adaptive = false, dt = dt, saveat = saveat);
reference_data = Array(full_CNODE(uv0, θ, st)[1]);
# And we unpack the solution to get the two species from
u = reshape(reference_data[1:(grid_GS.Nu), :, :],
    grid_GS.nux,
    grid_GS.nuy,
    size(reference_data, 2),
    :);
v = reshape(reference_data[(grid_GS.Nu + 1):end, :, :],
    grid_GS.nvx,
    grid_GS.nvy,
    size(reference_data, 2),
    :);

# ### Plot the data of the exact solution in the *fine* grid (DNS)
using Plots, Plots.PlotMeasures
anim = Animation()
fig = plot(layout = (3, 2), size = (600, 900))
@gif for i in 1:1000:size(u, 4)
    p1 = heatmap(u[:, :, 1, i],
        axis = false,
        cbar = false,
        aspect_ratio = 1,
        color = :reds,
        title = "u(x,y)")
    p2 = heatmap(v[:, :, 1, i],
        axis = false,
        cbar = false,
        aspect_ratio = 1,
        color = :blues,
        title = "v(x,y)")
    p3 = heatmap(u[:, :, 2, i], axis = false, cbar = false, aspect_ratio = 1, color = :reds)
    p4 = heatmap(v[:, :, 2, i],
        axis = false,
        cbar = false,
        aspect_ratio = 1,
        color = :blues)
    p5 = heatmap(u[:, :, 3, i], axis = false, cbar = false, aspect_ratio = 1, color = :reds)
    p6 = heatmap(v[:, :, 3, i],
        axis = false,
        cbar = false,
        aspect_ratio = 1,
        color = :blues)
    time = round(i * saveat, digits = 0)
    fig = plot(p1, p2, p3, p4, p5, p6, layout = (3, 2),
        plot_title = "time = $(time)", margin = 0mm)
    frame(anim, fig)
end
if isdir("./plots")
    gif(anim, "./plots/02.04-DNS.gif", fps = 10)
else
    gif(anim, "examples/plots/02.04-DNS.gif", fps = 10)
end

# ## Exact solution in a *coarse* grid for $v$ - LES
# Let's solve the model in a *coarse* grid for $v$.
# ### Collect the LES data: *coarse* grid
# We redefine the grid parameters in order to have a coarser grid for $v$.
nvx = nvy = 30
dvx = nux * dux / nvx
dvy = nuy * duy / nvy
coarse_grid = Grid(dux, duy, nux, nuy, dvx, dvy, nvx, nvy);

# Let's see what happens if we use a non-closed model on a smaller grid
# (we are not using any NN here).
using Images
f_coarse_CNODE = create_f_CNODE(F_u, G_v, coarse_grid; is_closed = false)
θ, st = Lux.setup(rng, f_coarse_CNODE);
uv0 = burnout_data[:, :, end];
u0b = reshape(uv0[1:(grid_GS.Nu), :], grid_GS.nux, grid_GS.nuy, :);
v0b = reshape(uv0[(grid_GS.Nu + 1):end, :], grid_GS.nvx, grid_GS.nvy, :);
v0_coarse = imresize(v0b, (coarse_grid.nvx, coarse_grid.nvy));
uv0_coarse = vcat(reshape(u0b, coarse_grid.Nu, :), reshape(v0_coarse, coarse_grid.Nv, :));
# ### Define model without closure: LES
les_CNODE = NeuralODE(f_coarse_CNODE,
    trange,
    solver_algo,
    adaptive = false,
    dt = dt,
    saveat = saveat);
les_solution = Array(les_CNODE(uv0_coarse, θ, st)[1]);
u_les = reshape(les_solution[1:(coarse_grid.Nu), :, :],
    coarse_grid.nux,
    coarse_grid.nuy,
    size(les_solution, 2),
    :);
v_les = reshape(les_solution[(coarse_grid.Nu + 1):end, :, :],
    coarse_grid.nvx,
    coarse_grid.nvy,
    size(les_solution, 2),
    :);
# ### Plot the results of LES (no closure)
anim = Animation()
fig = plot(layout = (3, 5), size = (500, 300))
@gif for i in 1:1000:size(u_les, 4)
    p1 = heatmap(u_les[:, :, 1, i],
        axis = false,
        cbar = false,
        aspect_ratio = 1,
        color = :reds,
        title = "u(x,y) [C]")
    p2 = heatmap(v_les[:, :, 1, i],
        axis = false,
        cbar = false,
        aspect_ratio = 1,
        color = :blues,
        title = "v(x,y) [C]")
    p3 = heatmap(u[:, :, 1, i],
        axis = false,
        cbar = false,
        aspect_ratio = 1,
        color = :reds,
        title = "u(x,y) [F]")
    p4 = heatmap(v[:, :, 1, i],
        axis = false,
        cbar = false,
        aspect_ratio = 1,
        color = :blues,
        title = "v(x,y) [F]")
    e = u_les[:, :, 1, i] .- u[:, :, 1, i]
    p5 = heatmap(e,
        axis = false,
        cbar = false,
        aspect_ratio = 1,
        color = :greens,
        title = "u-Diff")
    p6 = heatmap(u_les[:, :, 2, i],
        axis = false,
        cbar = false,
        aspect_ratio = 1,
        color = :reds)
    p7 = heatmap(v_les[:, :, 2, i],
        axis = false,
        cbar = false,
        aspect_ratio = 1,
        color = :blues)
    p8 = heatmap(u[:, :, 2, i], axis = false, cbar = false, aspect_ratio = 1, color = :reds)
    p9 = heatmap(v[:, :, 2, i],
        axis = false,
        cbar = false,
        aspect_ratio = 1,
        color = :blues)
    e = u_les[:, :, 2, i] .- u[:, :, 2, i]
    p10 = heatmap(e,
        axis = false,
        cbar = false,
        aspect_ratio = 1,
        color = :greens,
        title = "u-Diff")
    p11 = heatmap(u_les[:, :, 3, i],
        axis = false,
        cbar = false,
        aspect_ratio = 1,
        color = :reds)
    p12 = heatmap(v_les[:, :, 3, i],
        axis = false,
        cbar = false,
        aspect_ratio = 1,
        color = :blues)
    p13 = heatmap(u[:, :, 3, i],
        axis = false,
        cbar = false,
        aspect_ratio = 1,
        color = :reds)
    p14 = heatmap(v[:, :, 3, i],
        axis = false,
        cbar = false,
        aspect_ratio = 1,
        color = :blues)
    e = u_les[:, :, 3, i] .- u[:, :, 3, i]
    p15 = heatmap(e,
        axis = false,
        cbar = false,
        aspect_ratio = 1,
        color = :greens,
        title = "u-Diff")
    time = round(i * saveat, digits = 0)
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
        p13,
        p14,
        p15,
        layout = (3, 5),
        plot_title = "time = $(time)",
        margin = 0mm)
    frame(anim, fig)
end
if isdir("./plots")
    gif(anim, "./plots/02.04-LES.gif", fps = 10)
else
    gif(anim, "examples/plots/02.04-LES.gif", fps = 10)
end

# ## Closure model + LES 
# Now let's train a Neural Network closure that added to a *coarse* grid solution of $v$ (LES) would approximate the *fine* gride results (DNS).
# ### Prepare reference data (labels)
# In order to prepare the loss function, we compute from the simulation data the target that we would like to fit. In the example $u$ will be unchanged, while $v$ will be rescaled to the coarse grid.
u_target = reshape(reference_data[1:(grid_GS.Nu), :, :],
    grid_GS.nux,
    grid_GS.nuy,
    size(reference_data, 2),
    :);
v = reshape(reference_data[(grid_GS.Nu + 1):end, :, :],
    grid_GS.nvx,
    grid_GS.nvy,
    size(reference_data, 2),
    :);
v_target = imresize(v, (coarse_grid.nvx, coarse_grid.nvy));
# Pack $u$ and $v$ together in a `target` array where $u$ and $v$ are linearized in the first dimension.
target = vcat(reshape(u_target, grid_GS.Nu, size(u_target, 3), :),
    reshape(v_target, coarse_grid.Nv, size(v_target, 3), :));

# ### Closure Model
# Let's create the CNODE with the Neural Network closure. In this case we are going to use two different neural netwroks for the two components $u$ and $v$.
import CoupledNODE: create_fno_model
ch_fno = [2, 5, 5, 5, 2];
kmax_fno = [8, 8, 8, 8];
σ_fno = [Lux.gelu, Lux.gelu, Lux.gelu, identity];
NN_u = create_fno_model(kmax_fno, ch_fno, σ_fno);
NN_v = create_fno_model(kmax_fno, ch_fno, σ_fno);
f_closed_CNODE = create_f_CNODE(F_u, G_v, coarse_grid, NN_u, NN_v; is_closed = true);

## Get the model parameters
θ, st = Lux.setup(rng, f_closed_CNODE);

# ### Design the **loss function**
# For this example, we use *multishooting a posteriori* fitting (MulDtO), where we tell `Zygote` to compare `nintervals` of length `nunroll` to get the gradient. Notice that this method is differentiating through the solution of the NODE!
nunroll = 5
nintervals = 20
nsamples = 2
# We also define this auxiliary NODE `training_CNODE` that will be used for training so that we can use smaller time steps for the training.
dt_train = 0.05
# but we have to sample at the same rate as the data
saveat_train = saveat
t_train_range = (0.0f0, saveat_train * nunroll) # it has to be as long as nunroll
training_CNODE = NeuralODE(f_closed_CNODE,
    t_train_range,
    solver_algo,
    adaptive = false,
    dt = dt_train,
    saveat = saveat_train);

# Create the loss
import CoupledNODE: create_randloss_MulDtO
myloss = create_randloss_MulDtO(target,
    training_CNODE,
    st,
    nunroll = nunroll,
    nintervals = nintervals,
    nsamples = nsamples,
    λ_c = 1e2,
    λ_l1 = 1e-1);

## Initialize `lhist` to monitor the training loss.
lhist = Float32[];
## Initialize and trigger the compilation of the model
import ComponentArrays
pinit = ComponentArrays.ComponentArray(θ);
myloss(pinit)  # trigger compilation
# [!] Check that the loss does not get type warnings, otherwise it will be slower

# ### Transform the NeuralODE into an optimization problem
import OptimizationOptimisers: Optimization, OptimiserChain, Adam, ClipGrad
# * Select the autodifferentiation type
adtype = Optimization.AutoZygote();
optf = Optimization.OptimizationFunction((x, p) -> myloss(x), adtype);
optprob = Optimization.OptimizationProblem(optf, pinit);

# * Select the training algorithm
# We choose Adam with learning rate 0.1, with gradient clipping
ClipAdam = OptimiserChain(Adam(1.0f-2), ClipGrad(1));

# ### Train the CNODE
# (The block can be repeated to continue training)
import CoupledNODE: callback
result_neuralode = Optimization.solve(optprob,
    ClipAdam;
    callback = callback,
    maxiters = 3);
pinit = result_neuralode.u;
θ = pinit;
optprob = Optimization.OptimizationProblem(optf, pinit);

# *Note:* the training is rather slow, so realistically here we can not expect good results in a few iterations.

# ## Analyse the results
# Let's use the trained CNODE to compare the solution with the target.
trange = (0.0f0, 300.0f0)
trained_CNODE = NeuralODE(f_closed_CNODE,
    trange,
    solver_algo,
    adaptive = false,
    dt = dt,
    saveat = saveat);
trained_CNODE_solution = Array(trained_CNODE(uv0_coarse[:, 1:3], θ, st)[1]);
u_trained = reshape(trained_CNODE_solution[1:(coarse_grid.Nu), :, :],
    coarse_grid.nux,
    coarse_grid.nuy,
    size(trained_CNODE_solution, 2),
    :);
v_trained = reshape(trained_CNODE_solution[(coarse_grid.Nu + 1):end, :, :],
    coarse_grid.nvx,
    coarse_grid.nvy,
    size(trained_CNODE_solution, 2),
    :);

anim = Animation()
fig = plot(layout = (2, 5), size = (750, 300))
@gif for i in 1:40:size(u_trained, 4)
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
    p3 = heatmap(u_trained[:, :, 1, i],
        axis = false,
        cbar = false,
        aspect_ratio = 1,
        color = :reds,
        title = "Trained")
    p4 = heatmap(v_trained[:, :, 1, i],
        axis = false,
        cbar = false,
        aspect_ratio = 1,
        color = :blues)
    et = abs.(u[:, :, 1, i] .- u_trained[:, :, 1, i])
    p5 = heatmap(et,
        axis = false,
        cbar = false,
        aspect_ratio = 1,
        color = :greens,
        title = "Diff-u")
    p6 = heatmap(u[:, :, 2, i], axis = false, cbar = false, aspect_ratio = 1, color = :reds)
    p7 = heatmap(v[:, :, 2, i],
        axis = false,
        cbar = false,
        aspect_ratio = 1,
        color = :blues)
    p8 = heatmap(u_trained[:, :, 2, i],
        axis = false,
        cbar = false,
        aspect_ratio = 1,
        color = :reds)
    p9 = heatmap(v_trained[:, :, 2, i],
        axis = false,
        cbar = false,
        aspect_ratio = 1,
        color = :blues)
    e = abs.(u[:, :, 2, i] .- u_trained[:, :, 2, i])
    p10 = heatmap(e, axis = false, cbar = false, aspect_ratio = 1, color = :greens)

    time = round(i * saveat, digits = 0)
    fig = plot(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, layout = (2, 5), margin = 0mm)

    frame(anim, fig)
end
if isdir("./plots")
    gif(anim, "./plots/02.04-NNclosure.gif", fps = 10)
else
    gif(anim, "examples/plots/02.04-NNclosure.gif", fps = 10)
end
