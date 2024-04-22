const ArrayType = Array
import DifferentialEquations: Tsit5
const solver_algo = Tsit5()
const MY_TYPE = Float32 # use float32 if you plan to use a GPU
import CUDA # Test if CUDA is running
if CUDA.functional()
    CUDA.allowscalar(false)
    const ArrayType = CuArray
    import DiffEqGPU: GPUTsit5
    const solver_algo = GPUTsit5()
end

# # 2D Burgers equations
# In this example, we will solve the Burgers equation in 2D using the Neural ODEs framework. The Burgers equation is a fundamental equation in fluid dynamics and is given by:
# $$
# \begin{equation}
# \frac{\partial u}{\partial t} = - u \frac{\partial u}{\partial x} - v \frac{\partial u}{\partial y} + \nu \Delta u
# \frac{\partial v}{\partial t} = - u \frac{\partial v}{\partial x} - v \frac{\partial v}{\partial y} + \nu \Delta v
# \end{equation}
# $$
# where $\bm{u} = \left\{u(x,y,t), v(x,y,t)\right\}$ is the velocity field, $\nu$ is the viscosity coefficient, and ($x$,$y$) and $t$ are the spatial and temporal coordinates, respectively. 
# The equation is a non-linear partial differential equation that describes the evolution of a fluid flow in two spatial dimensions. The equation is named after Johannes Martinus Burgers, who introduced it in 1948 as a simplified model for turbulence.

# We start by defining the right-hand side of the Burgers equation. We will use the finite difference method to compute the spatial derivatives. 
# So the first step is to define the grid that we are going to use
import CoupledNODE: Grid
nux = nuy = nvx = nvy = 100
dux = 2π / nux
duy = 2π / nuy
dvx = 2π / nvx
dvy = 2π / nvy
grid_u = Grid(dim = 2, dx = dux, dy = duy, nx = nux, ny = nuy);
grid_v = Grid(dim = 2, dx = dvx, dy = dvy, nx = nvx, ny = nvy);

# The following function constructs the right-hand side of the Burgers equation:
import CoupledNODE: Laplacian, first_derivatives
using Zygote
function create_burgers_rhs(grids, force_params)
    ν = force_params[1]

    function Force(u, v)
        du_dx, du_dy = first_derivatives(u, grids[1].dx, grids[1].dy)
        dv_dx, dv_dy = first_derivatives(v, grids[2].dx, grids[2].dy)
        F = Zygote.@ignore -u .* du_dx - v .* du_dy .+
                           ν * Laplacian(u, grids[1].dx^2, grids[1].dy^2)
        G = Zygote.@ignore -u .* dv_dx - v .* dv_dy .+
                           ν * Laplacian(v, grids[2].dx^2, grids[2].dy^2)
        return F, G
    end
    return Force
end
# Notice that compared to the Gray-Scott example we are returning a single function that computes both components of the force at the same time. This is because the Burgers equation is a system of two coupled PDEs so we want to avoid recomputing the derivatives a second time.

# Let's set the parameters for the Burgers equation
ν = 0.005f0
# and we pack them into a tuple for the rhs Constructor
force_params = (ν,)
# we also need to pack the grids into a tuple
grid_B = (grid_u, grid_v)

# Now we can create the right-hand side of the NODE
FG = create_burgers_rhs(grid_B, force_params)

include("./../coupling_functions/functions_NODE.jl")
f_CNODE = create_f_CNODE(create_burgers_rhs, force_params, grid_B; is_closed = false);
import Random, LuxCUDA, Lux
rng = Random.seed!(1234)
θ, st = Lux.setup(rng, f_CNODE);

# Now we create the initial condition for the Burgers equation. 
# We start defining a gaussian pulse centered in the grid.:
function initialize_uv_gaussian(grids, u_bkg, v_bkg, sigma)
    u_initial = zeros(MY_TYPE, grids[1].nx, grids[1].ny)
    v_initial = zeros(MY_TYPE, grids[2].nx, grids[2].ny)
    # Create a Gaussian pulse centered in the grid
    for i in 1:(grids[2].nx)
        for j in 1:(grids[2].ny)
            x = i - grids[2].nx / 2
            y = j - grids[2].ny / 2
            v_initial[i, j] = v_bkg * exp(-(x^2 + y^2) / (2 * sigma^2))
            u_initial[i, j] = u_bkg * exp(-(x^2 + y^2) / (2 * sigma^2))
        end
    end

    return u_initial, v_initial
end

u_initial, v_initial = initialize_uv_gaussian(grid_B, 2.0f0, 2.0f0, 20);
# We can now define the initial condition as a flattened concatenated array
uv0 = vcat(reshape(u_initial, grid_B[1].nx * grid_B[1].ny, 1),
    reshape(v_initial, grid_B[2].nx * grid_B[2].ny, 1))

# test the force
FG(u_initial, v_initial)

# The first phase of the Burger solution will be the formation of the shock. We use a small time step to resolve the shock formation.
import DiffEqFlux: NeuralODE
t_shock = 2.5f0
dt_shock = 0.005f0
trange_burn = (0.0f0, t_shock)
saveat_shock = 0.01f0
shock_CNODE = NeuralODE(f_CNODE,
    trange_burn,
    solver_algo,
    adaptive = false,
    dt = dt_shock,
    saveat = saveat_shock);
shock_CNODE_solution = Array(shock_CNODE(uv0, θ, st)[1])
uv_shock = shock_CNODE_solution[:, :, 1, end];

# And we unpack the solution to get the two species from
u_shock = reshape(shock_CNODE_solution[1:(grid_B[1].N), :, :],
    grid_B[1].nx,
    grid_B[1].ny,
    size(shock_CNODE_solution, 2),
    :)
v_shock = reshape(shock_CNODE_solution[(grid_B[1].N + 1):end, :, :],
    grid_B[2].nx,
    grid_B[2].ny,
    size(shock_CNODE_solution, 2),
    :);

# Plot 
x = range(0, 2π, length = grid_B[1].nx)
y = range(0, 2π, length = grid_B[1].ny)
using Plots #, Plotly
anim = Animation()
fig = plot(layout = (1, 2), size = (400, 800))
@gif for i in 1:2:size(u_shock, 4)
    p1 = surface(x, y, u_shock[:, :, 1, i], color = :viridis, cbar = false, xlabel = "x",
        ylabel = "y", zlabel = "u", title = "u field", camera = (45, 45))
    p2 = surface(x, y, v_shock[:, :, 1, i], color = :viridis, cbar = false, xlabel = "x",
        ylabel = "y", zlabel = "v", title = "v field", camera = (45, 45))

    title = "Time $(round(i*saveat_shock, digits=2))"
    fig = plot(p1, p2, layout = (1, 2), title = title)
    frame(anim, fig)
end

# Then there is a phase of shock dissipation
t_diss = 35.0f0
dt_diss = 0.01f0
trange_burn = (0.0f0, t_diss)
saveat_diss = 0.4f0
diss_CNODE = NeuralODE(f_CNODE,
    trange_burn,
    solver_algo,
    adaptive = false,
    dt = dt_diss,
    saveat = saveat_diss);
diss_CNODE_solution = Array(diss_CNODE(uv_shock, θ, st)[1])
uv_diss = diss_CNODE_solution[:, :, end];
u_diss = reshape(diss_CNODE_solution[1:(grid_B[1].N), :, :],
    grid_B[1].nx,
    grid_B[1].ny,
    size(diss_CNODE_solution, 2),
    :)
v_diss = reshape(diss_CNODE_solution[(grid_B[1].N + 1):end, :, :],
    grid_B[2].nx,
    grid_B[2].ny,
    size(diss_CNODE_solution, 2),
    :);
anim = Animation()
fig = plot(layout = (1, 2), size = (400, 800))
@gif for i in 1:2:size(u_diss, 4)
    p1 = surface(x, y, u_diss[:, :, 1, i], color = :viridis, cbar = false, xlabel = "x",
        ylabel = "y", zlabel = "u", title = "u field", camera = (45, 45))
    p2 = surface(x, y, v_diss[:, :, 1, i], color = :viridis, cbar = false, xlabel = "x",
        ylabel = "y", zlabel = "v", title = "v field", camera = (45, 45))

    title = "Time $(round(i*saveat_diss, digits=2))"
    fig = plot(p1, p2, layout = (1, 2), title = title)
    frame(anim, fig)
end

# And then the Burgers equation reaches a steady state
t_steady = 150.0f0
dt_steady = 0.1f0
trange_burn = (0.0f0, 100.0f0)
saveat_steady = 2.0f0
steady_CNODE = NeuralODE(f_CNODE,
    trange_burn,
    solver_algo,
    adaptive = false,
    dt = dt_steady,
    saveat = saveat_steady);
steady_CNODE_solution = Array(steady_CNODE(uv_diss, θ, st)[1])
uv_steady = steady_CNODE_solution[:, :, end];
u_steady = reshape(steady_CNODE_solution[1:(grid_B[1].N), :, :],
    grid_B[1].nx,
    grid_B[1].ny,
    size(steady_CNODE_solution, 2),
    :)
v_steady = reshape(steady_CNODE_solution[(grid_B[1].N + 1):end, :, :],
    grid_B[2].nx,
    grid_B[2].ny,
    size(steady_CNODE_solution, 2),
    :);
anim = Animation()
fig = plot(layout = (1, 2), size = (400, 800))
@gif for i in 1:2:size(u_steady, 4)
    p1 = surface(x, y, u_steady[:, :, 1, i], color = :viridis, cbar = false, xlabel = "x",
        ylabel = "y", zlabel = "u", title = "u field", camera = (45, 45))
    p2 = surface(x, y, v_steady[:, :, 1, i], color = :viridis, cbar = false, xlabel = "x",
        ylabel = "y", zlabel = "v", title = "v field", camera = (45, 45))

    title = "Time $(round(i*saveat_steady, digits=2))"
    fig = plot(p1, p2, layout = (1, 2), title = title)
    frame(anim, fig)
end

# Now plot the whole trajectory
u_total = cat(u_shock, u_diss, u_steady, dims = 4)
v_total = cat(v_shock, v_diss, v_steady, dims = 4)
t_total = vcat(0:saveat_shock:t_shock, t_shock:saveat_diss:(t_shock + t_diss),
    (t_shock + t_diss):saveat_steady:(t_shock + t_diss + t_steady))
label = ["Shock" for i in 1:size(u_shock, 4)]
append!(label, ["Dissipation" for i in 1:size(u_diss, 4)])
append!(label, ["Steady" for i in 1:size(u_steady, 4)])
anim = Animation()
fig = plot(layout = (1, 2), size = (400, 800))
@gif for i in 1:2:size(u_total, 4)
    p1 = surface(x, y, u_total[:, :, 1, i], color = :viridis, cbar = false, xlabel = "x",
        ylabel = "y", zlabel = "u", title = "u field", camera = (45, 45))
    p2 = surface(x, y, v_total[:, :, 1, i], color = :viridis, cbar = false, xlabel = "x",
        ylabel = "y", zlabel = "v", title = "v field", camera = (45, 45))

    title = "Time $(round(t_total[i], digits=2)) $(label[i])"
    fig = plot(p1, p2, layout = (1, 2), title = title)
    frame(anim, fig)
end
if isdir("./plots")
    gif(anim, "plots/04.01_2DBurgers.gif", fps = 8)
else
    gif(anim, "examples/plots/04.01_2DBurgers.gif", fps = 8)
end
