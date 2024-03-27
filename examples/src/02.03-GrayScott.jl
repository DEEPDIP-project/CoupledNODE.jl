# # Learning the Gray-Scott model: Effect of grid coarsening
# In this example we want to show the effect of grid coarsening on the solution of a PDE.
# We will introduce one of the most important problems in the numerical solution of PDEs, that we will try to solve in the following examples using CNODEs.

# We use again the GS model, which is defined from
# \begin{equation}\begin{cases} \frac{du}{dt} = D_u \Delta u - uv^2 + f(1-u)  \equiv F_u(u,v) \\ \frac{dv}{dt} = D_v \Delta v + uv^2 - (f+k)v  \equiv G_v(u,v)\end{cases} \end{equation}
# where $u(x,y,t):\mathbb{R}^2\times \mathbb{R}\rightarrow \mathbb{R}$ is the concentration of species 1, while $v(x,y,t)$ is the concentration of species two. This model reproduce the effect of the two species diffusing in their environment, and reacting together.
# This effect is captured by the ratios between $D_u$ and $D_v$ (the diffusion coefficients) and $f$ and $k$ (the reaction rates).

# ### The *exact* solution
# Even if the GS model can not be solved analytically, we can discretize it on a very fine grid and expect its solution to be almost exact. 
# We will use it as a reference to compare the solution on a coarser grid.
# Notice that the simple fact that we are discretizing, makes this solution technically a DNS (Direct Numerical Simulation) and not an exact solution, but since we are using a very fine grid, we will call it *exact* for simplicity.

# Let's define the finest grid using 200 steps of 0.5 in each direction, reaching a 100[L] x 100[L] domain.
import CoupledNODE: Grid
dux = duy = dvx = dvy = 0.5
nux = nuy = nvx = nvy = 200
grid_GS = Grid(dux, duy, nux, nuy, dvx, dvy, nvx, nvy);

# We start with a central concentration of $v$
function initialize_uv(grid, u_bkg, v_bkg, center_size)
    u_initial = u_bkg * ones(grid.nux, grid.nuy)
    v_initial = zeros(grid.nvx, grid.nvy)
    v_initial[Int(grid.nvx / 2 - center_size):Int(grid.nvx / 2 + center_size), Int(grid.nvy / 2 - center_size):Int(grid.nvy / 2 + center_size)] .= v_bkg
    return u_initial, v_initial
end
u_initial, v_initial = initialize_uv(grid_GS, 0.8, 0.9, 4);

# We can now define the initial condition as a flattened concatenated array
uv0 = vcat(reshape(u_initial, grid_GS.nux * grid_GS.nuy, 1),
    reshape(v_initial, grid_GS.nvx * grid_GS.nvy, 1))

# From the literature, we have selected the following parameters in order to form nice patterns
D_u = 0.16
D_v = 0.08
f = 0.055
k = 0.062;

# Here we (the user) define the **right hand sides** of the equations
import CoupledNODE: Laplacian
F_u(u, v, grid) = D_u * Laplacian(u, grid.dux, grid.duy) .- u .* v .^ 2 .+ f .* (1.0 .- u)
G_v(u, v, grid) = D_v * Laplacian(v, grid.dvx, grid.dvy) .+ u .* v .^ 2 .- (f + k) .* v

# Once the forces have been defined, we can create the CNODE
f_CNODE = create_f_CNODE(F_u, G_v, grid_GS; is_closed = false)
# and we ask Lux for the parameters to train and their structure [none in this example]
import Random, Lux
rng = Random.seed!(1234)
θ, st = Lux.setup(rng, f_CNODE);

# Actuallly, we are not training any parameters, but using `NeuralODE` for consistency with the resto of examples. Therefore, we see that $\theta$ is empty.
length(θ)

# We now do a short *burnout run* to get rid of the initial artifacts
import DifferentialEquations: Tsit5
import DiffEqFlux: NeuralODE
trange_burn = (0.0, 10.0)
dt_burn, saveat_burn = (1e-2, 1)
full_CNODE = NeuralODE(f_CNODE,
    trange_burn,
    Tsit5(),
    adaptive = false,
    dt = dt_burn,
    saveat = saveat_burn)
burnout_CNODE_solution = Array(full_CNODE(uv0, θ, st)[1])

# **CNODE run** 
# We use the output of the burnout to start a longer simulations
uv0 = burnout_CNODE_solution[:, :, end];
trange = (0.0, 7000.0)
dt, saveat = (0.5, 20)
full_CNODE = NeuralODE(f_CNODE, trange, Tsit5(), adaptive = false, dt = dt, saveat = saveat)
untrained_CNODE_solution = Array(full_CNODE(uv0, θ, st)[1])
# And we unpack the solution to get the two species from
u_exact = reshape(untrained_CNODE_solution[1:(grid_GS.Nu), :, :],
    grid_GS.nux,
    grid_GS.nuy,
    size(untrained_CNODE_solution, 2),
    :)
v_exact = reshape(untrained_CNODE_solution[(grid_GS.Nu + 1):end, :, :],
    grid_GS.nvx,
    grid_GS.nvy,
    size(untrained_CNODE_solution, 2),
    :);

# Plot the solution as an animation
using Plots
anim = Animation()
fig = plot(layout = (1, 2), size = (600, 300))
@gif for i in 1:2:size(u_exact, 4)
    p1 = heatmap(u_exact[:, :, 1, i],
        axis = false,
        bar = false,
        aspect_ratio = 1,
        color = :reds,
        title = "u(x,y)")
    p2 = heatmap(v_exact[:, :, 1, i],
        axis = false,
        bar = false,
        aspect_ratio = 1,
        color = :blues,
        title = "v(x,y)")
    time = round(i * saveat, digits = 0)
    fig = plot(p1, p2, layout = (1, 2), plot_title = "time = $(time)")
    frame(anim, fig)
end

# ### **DNS**: Direct Numerical Simulation
# The DNS is a refined solution of the PDE, where the grid is so fine that the solution is almost exact.
# Technically, the *exact solution* in the previous section is also a DNS (because it is discrete), but we want to show that the grid can be made a bit coarser while preserving the dynamics of the original PDE.
# This is because we want an efficient DNS that we can run to collect data for analysis, prediction and to train ML models.

# The DNS grid will consists of 150 steps of 100 in each direction, covering the 100[L] x 100[L] domain.
dux = duy = dvx = dvy = 100 / 150
nux = nuy = nvx = nvy = 150
dns_grid = Grid(dux, duy, nux, nuy, dvx, dvy, nvx, nvy);

# Use the same initial condition as the exact solution 
import Images: imresize
u0_dns = imresize(u_initial, (dns_grid.nux, dns_grid.nuy));
v0_dns = imresize(v_initial, (dns_grid.nvx, dns_grid.nvy));
uv0_dns = vcat(reshape(u0_dns, dns_grid.nux * dns_grid.nuy, 1),
    reshape(v0_dns, dns_grid.nvx * dns_grid.nvy, 1))

# define the forces and create the CNODE
f_dns = create_f_CNODE(F_u, G_v, dns_grid; is_closed = false)
θ, st = Lux.setup(rng, f_dns);

# burnout run
dns_CNODE = NeuralODE(f_dns,
    trange_burn,
    Tsit5(),
    adaptive = false,
    dt = dt_burn,
    saveat = saveat_burn)
burnout_dns = Array(dns_CNODE(uv0_dns, θ, st)[1])

# DNS simulation
uv0 = burnout_dns[:, :, end];
dns_CNODE = NeuralODE(f_dns, trange, Tsit5(), adaptive = false, dt = dt, saveat = saveat)
dns_solution = Array(dns_CNODE(uv0, θ, st)[1])
u_dns = reshape(dns_solution[1:(dns_grid.Nu), :, :],
    dns_grid.nux,
    dns_grid.nuy,
    size(dns_solution, 2),
    :)
v_dns = reshape(dns_solution[(dns_grid.Nu + 1):end, :, :],
    dns_grid.nvx,
    dns_grid.nvy,
    size(dns_solution, 2),
    :);

# Plot DNS vs exact solution
anim = Animation()
fig = plot(layout = (2, 2), size = (600, 600))
@gif for i in 1:2:size(u_exact, 4)
    p1 = heatmap(u_exact[:, :, 1, i],
        axis = false,
        bar = false,
        aspect_ratio = 1,
        color = :reds,
        title = "u(x,y)")
    p2 = heatmap(v_exact[:, :, 1, i],
        axis = false,
        bar = false,
        aspect_ratio = 1,
        color = :blues,
        title = "v(x,y)")
    p3 = heatmap(u_dns[:, :, 1, i],
        axis = false,
        bar = false,
        aspect_ratio = 1,
        color = :reds,
        title = "u(x,y)")
    p4 = heatmap(v_dns[:, :, 1, i],
        axis = false,
        bar = false,
        aspect_ratio = 1,
        color = :blues,
        title = "v(x,y)")
    time = round(i * saveat, digits = 0)
    fig = plot(p1, p2, p3, p4, layout = (2, 2), plot_title = "time = $(time)")
    frame(anim, fig)
end

# ### **LES**: Large Eddy Simulation
# The LES is a coarser solution of the PDE, where the grid is so coarse that the solution is not exact, but it still captures the main features of the original PDE.
# It is used to reduce the computational cost of the DNS such that we can run it for longer.
# However we will see that what it saves in computational cost, it loses in accuracy.
# In the following examples, the goal of the CNODE will be to correct the LES solution to make it more accurate.

# This is the grid we will use for the LES, with 75 steps of 100/75[L] in each direction, covering the 100[L] x 100[L] domain.
dux = duy = dvx = dvy = 100 / 75
nux = nuy = nvx = nvy = 75
les_grid = Grid(dux, duy, nux, nuy, dvx, dvy, nvx, nvy);

# Use the same initial condition as the exact solution
u0_les = imresize(u_initial, (les_grid.nux, les_grid.nuy));
v0_les = imresize(v_initial, (les_grid.nvx, les_grid.nvy));
uv0_les = vcat(reshape(u0_les, les_grid.nux * les_grid.nuy, 1),
    reshape(v0_les, les_grid.nvx * les_grid.nvy, 1))

# Compare the initial conditions
p1 = heatmap(u_initial,
    axis = false,
    cbar = false,
    aspect_ratio = 1,
    title = "u exact",
    color = :reds)
p2 = heatmap(v_initial,
    axis = false,
    cbar = false,
    aspect_ratio = 1,
    title = "v exact",
    color = :blues)
p3 = heatmap(u0_dns,
    axis = false,
    cbar = false,
    aspect_ratio = 1,
    title = "u_0 DNS",
    color = :reds)
p4 = heatmap(v0_dns,
    axis = false,
    cbar = false,
    aspect_ratio = 1,
    title = "v_0 DNS",
    color = :blues)
p5 = heatmap(u0_les,
    axis = false,
    cbar = false,
    aspect_ratio = 1,
    title = "u_0 LES",
    color = :reds)
p6 = heatmap(v0_les,
    axis = false,
    cbar = false,
    aspect_ratio = 1,
    title = "v_0 LES",
    color = :blues)
plot(p1, p2, p3, p4, p5, p6, layout = (3, 2), plot_title = "Initial conditions")

# define the forces and create the CNODE
f_les = create_f_CNODE(F_u, G_v, les_grid; is_closed = false)
θ, st = Lux.setup(rng, f_les);

# burnout run
les_CNODE = NeuralODE(f_les,
    trange_burn,
    Tsit5(),
    adaptive = false,
    dt = dt_burn,
    saveat = saveat_burn)
burnout_les = Array(les_CNODE(uv0_les, θ, st)[1])

# LES simulation
uv0 = burnout_les[:, :, end];
les_CNODE = NeuralODE(f_les, trange, Tsit5(), adaptive = false, dt = dt, saveat = saveat)
les_solution = Array(les_CNODE(uv0, θ, st)[1])
# And we unpack the solution to get the two species from
u_les = reshape(les_solution[1:(les_grid.Nu), :, :],
    les_grid.nux,
    les_grid.nuy,
    size(les_solution, 2),
    :)
v_les = reshape(les_solution[(les_grid.Nu + 1):end, :, :],
    les_grid.nvx,
    les_grid.nvy,
    size(les_solution, 2),
    :);

anim = Animation()
fig = plot(layout = (3, 2), size = (600, 900))
@gif for i in 1:2:size(u_exact, 4)
    p1 = heatmap(u_exact[:, :, 1, i],
        axis = false,
        cbar = false,
        aspect_ratio = 1,
        color = :reds,
        title = "u exact")
    p2 = heatmap(v_exact[:, :, 1, i],
        axis = false,
        cbar = false,
        aspect_ratio = 1,
        color = :blues,
        title = "v exact")
    p3 = heatmap(u_dns[:, :, 1, i],
        axis = false,
        cbar = false,
        aspect_ratio = 1,
        color = :reds,
        title = "u DNS")
    p4 = heatmap(v_dns[:, :, 1, i],
        axis = false,
        cbar = false,
        aspect_ratio = 1,
        color = :blues,
        title = "v DNS")
    p5 = heatmap(u_les[:, :, 1, i],
        axis = false,
        cbar = false,
        aspect_ratio = 1,
        color = :reds,
        title = "u LES")
    p6 = heatmap(v_les[:, :, 1, i],
        axis = false,
        cbar = false,
        aspect_ratio = 1,
        color = :blues,
        title = "v LES")
    time = round(i * saveat, digits = 0)
    fig = plot(p1, p2, p3, p4, p5, p6, layout = (3, 2), plot_title = "time = $(time)")
    frame(anim, fig)
end
if isdir("./plots")
    gif(anim, "plots/02.03_gridsize.gif", fps = 10)
else
    gif(anim, "examples/plots/02.03_gridsize.gif", fps = 10)
end
# In the figure we see that the LES has induced some artifacts that influences the dynamics. In the next example, we will solve these artifacts using the Neural part of the CNODEs.
