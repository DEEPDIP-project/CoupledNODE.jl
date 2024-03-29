# # Gray-Scott model - explicit solution
# In following examples we will use the GS model to showcase how it can be represented as Coupled Neural ODEs (CNODEs). But let us first explore the GS model starting with an explicit solution of it. We will be using [SciML](https://sciml.ai/) package [DiffEqFlux.jl`](https://github.com/SciML/DiffEqFlux.jl) and scpecifically [`NeuralODE`](https://docs.sciml.ai/DiffEqFlux/stable/examples/neural_ode/) for defining and solving the problem.

# The system that we want to solve, called the the Gray-Scott model, is defined by the following equations:
# \begin{equation}\begin{cases} \frac{du}{dt} = D_u \Delta u - uv^2 + f(1-u)  \equiv F_u(u,v) \\ \frac{dv}{dt} = D_v \Delta v + uv^2 - (f+k)v  \equiv G_v(u,v)\end{cases} \end{equation}
# where $u(x,y,t):\mathbb{R}^2\times \mathbb{R}\rightarrow \mathbb{R}$ is the concentration of species 1, while $v(x,y,t)$ is the concentration of species 2. This model reproduce the effect of the two species diffusing in their environment and reacting together.
# This effect is captured by the ratios between $D_u$ and $D_v$ (the diffusion coefficients) and $f$ and $k$ (the reaction rates).

# Let's start creating a grid to discretize the problem. Notice that in literature the coefficients are usually scaled such that $dx=dy=1$, so we will use this scaling to have a direct comparison with literature.
import CoupledNODE: Grid
dux = duy = dvx = dvy = 1.0
nux = nuy = nvx = nvy = 100
grid_GS = Grid(dux, duy, nux, nuy, dvx, dvy, nvx, nvy);

# We start defining a central concentration of $v$ and a constant concentration of $u$:
function initialize_uv(grid, u_bkg, v_bkg, center_size)
    u_initial = u_bkg * ones(grid.nux, grid.nuy)
    v_initial = zeros(grid.nvx, grid.nvy)
    v_initial[Int(grid.nvx / 2 - center_size):Int(grid.nvx / 2 + center_size), Int(grid.nvy / 2 - center_size):Int(grid.nvy / 2 + center_size)] .= v_bkg
    return u_initial, v_initial
end
u_initial, v_initial = initialize_uv(grid_GS, 0.8, 0.9, 4);

# We can now define the initial condition as a flattened concatenated array
uv0 = vcat(reshape(u_initial, grid_GS.nux * grid_GS.nuy, 1),
    reshape(v_initial, grid_GS.nvx * grid_GS.nvy, 1));

# From the literature, we select the following parameters in order to form nice patterns.
D_u = 0.16
D_v = 0.08
f = 0.055
k = 0.062;

# Define the **right hand sides** of the two equations:
import CoupledNODE: Laplacian
function F_u(u, v, grid_GS)
    D_u * Laplacian(u, grid_GS.dux, grid_GS.duy) .- u .* v .^ 2 .+ f .* (1.0 .- u)
end
function G_v(u, v, grid_GS)
    D_v * Laplacian(v, grid_GS.dvx, grid_GS.dvy) .+ u .* v .^ 2 .- (f + k) .* v
end

# Once the functions have been defined, we can create the CNODE
# Notice that in the future, this same constructor will be able to use the user provided neural network to close the equations
import CoupledNODE: create_f_CNODE
f_CNODE = create_f_CNODE(F_u, G_v, grid_GS; is_closed = false);
# and we ask Lux for the parameters to train and their structure
import Lux, Random
rng = Random.seed!(1234);
θ, st = Lux.setup(rng, f_CNODE);
# in this example we are not training any parameters, so we can confirm that the vector θ is empty
length(θ) == 0;

# We now do a short *burnout run* to get rid of the initial artifacts. This allows us to discard the transient dynamics and to have a good initial condition for the data collection run.

using DifferentialEquations: Tsit5
using DiffEqFlux: NeuralODE
trange_burn = (0.0, 10.0);
dt, saveat = (1e-2, 1);
full_CNODE = NeuralODE(f_CNODE,
    trange_burn,
    Tsit5(),
    adaptive = false,
    dt = dt,
    saveat = saveat);
burnout_CNODE_solution = Array(full_CNODE(uv0, θ, st)[1]);

# **CNODE run**

# We use the output of the *burnout run* to start a longer simulation
uv0 = burnout_CNODE_solution[:, :, end];
trange = (0.0, 8000.0);
# the maximum suggested time step for GS is defined as `1/(4 * Dmax)`
dt, saveat = (1 / (4 * max(D_u, D_v)), 25);
full_CNODE = NeuralODE(f_CNODE, trange, Tsit5(), adaptive = false, dt = dt, saveat = saveat);
untrained_CNODE_solution = Array(full_CNODE(uv0, θ, st)[1]);
# And we unpack the solution to get the two species. Remember that we have concatenated $u$ and $v$ in the same array.
u = reshape(untrained_CNODE_solution[1:(grid_GS.Nu), :, :],
    grid_GS.nux,
    grid_GS.nuy,
    size(untrained_CNODE_solution, 2),
    :);
v = reshape(untrained_CNODE_solution[(grid_GS.Nu + 1):end, :, :],
    grid_GS.nvx,
    grid_GS.nvy,
    size(untrained_CNODE_solution, 2),
    :);

# Finally, plot the solution as an animation
using Plots
anim = Animation()
fig = plot(layout = (1, 2), size = (600, 300))
@gif for i in 1:2:size(u, 4)
    p1 = heatmap(u[:, :, 1, i],
        axis = false,
        bar = true,
        aspect_ratio = 1,
        color = :reds,
        title = "u(x,y)")
    p2 = heatmap(v[:, :, 1, i],
        axis = false,
        bar = true,
        aspect_ratio = 1,
        color = :blues,
        title = "v(x,y)")
    time = round(i * saveat, digits = 0)
    fig = plot(p1, p2, layout = (1, 2), plot_title = "time = $(time)")
    frame(anim, fig)
end
if isdir("./plots")
    gif(anim, "./plots/02.00.GS.gif", fps = 10)
else
    gif(anim, "examples/plots/02.00.GS.gif", fps = 10)
end
