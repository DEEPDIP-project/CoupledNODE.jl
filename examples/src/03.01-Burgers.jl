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

# # Burgers equations
# In this example, we will solve the Burgers equation in using the Neural ODEs framework. The Burgers equation is a fundamental equation in fluid dynamics and is given by:
# \begin{equation}
# \frac{\partial u}{\partial t} = - u \frac{\partial u}{\partial x} + \nu \frac{\partial u^2}{\partial x^2}
# \end{equation}
# where $u(x,t)$ is the velocity of the fluid, $\nu$ is the viscosity coefficient, and x$ and $t$ are the spatial and temporal coordinates, respectively. The equation is a non-linear partial differential equation that describes the evolution of a fluid flow in one spatial dimensions. The equation is named after Johannes Martinus Burgers, who introduced it in 1948 as a simplified model for turbulence.

# We start by defining the right-hand side of the Burgers equation. We will use the finite difference method to compute the spatial derivatives. 
# So the first step is to define the grid that we are going to use
import CoupledNODE: Grid
nux = 100
dux = 2π / nux
xgrid_B = Grid(dim = 1, dx = dux, nx = nux)

# The following function constructs the right-hand side of the Burgers equation:
import CoupledNODE: Laplacian, first_derivatives
using Zygote
function create_burgers_rhs(grids, force_params)
    ν = force_params[1]

    function Force(u)
        du_dx = first_derivatives(u, grids[1].dx)
        F = Zygote.@ignore -u .* du_dx +
                           ν * Laplacian(u, grids[1].dx^2)
        return F
    end
    return Force
end

# Let's set the parameters for the Burgers equation
ν = 0.01f0
# and we pack them into a tuple for the rhs Constructor
force_params = (ν,)
# we also need to pack the grid into a tuple
grid_B = (xgrid_B,)

# Now we can create the right-hand side of the NODE
F = create_burgers_rhs(grid_B, force_params)

# We can now generate the initial conditions for the Burgers equation. We will use a combination of sine waves and noise to create the initial conditions.
function generate_initial_conditions(n_samples::Int)
    x = range(0, stop = 2π, length = nux)
    freqs = [2π, 4π, 6π]  # Frequencies of sine waves
    amplitudes = [1.0, 0.5, 0.3]  # Amplitudes of sine waves

    u0_list = Array{Float32, 2}(undef, nux, n_samples)

    for j in 1:n_samples
        u0 = @view u0_list[:, j]
        u0 .= 0.0
        for i in 1:length(freqs)
            u0 .+= amplitudes[i] * sin.(freqs[i] * x)
        end
        # Add random noise
        u0 .+= 0.1 * randn(nux)
    end
    return u0_list
end

u_0 = generate_initial_conditions(4)

x = range(0, stop = 2π, length = nux)
using Plots
#plot(x, u_0, label = ["Sample 1" "Sample 2" "Sample 3" "Sample 4"], xlabel = "x", ylabel = "u", title = "Initial conditions for the Burgers equation")

F(u_0)

include("./../coupling_functions/functions_NODE.jl")
f_CNODE = create_f_CNODE(create_burgers_rhs, force_params, grid_B; is_closed = false);
import Random, LuxCUDA, Lux
rng = Random.seed!(1234)
θ, st = Lux.setup(rng, f_CNODE);

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
u_shock = reshape(shock_CNODE_solution[1:(grid_B.Nu), :, :],
    grid_B.nux,
    grid_B.nuy,
    size(shock_CNODE_solution, 2),
    :)
v_shock = reshape(shock_CNODE_solution[(grid_B.Nu + 1):end, :, :],
    grid_B.nvx,
    grid_B.nvy,
    size(shock_CNODE_solution, 2),
    :);

# Plot 
x = range(0, 2π, length = grid_B.nux)
y = range(0, 2π, length = grid_B.nuy)
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
u_diss = reshape(diss_CNODE_solution[1:(grid_B.Nu), :, :],
    grid_B.nux,
    grid_B.nuy,
    size(diss_CNODE_solution, 2),
    :)
v_diss = reshape(diss_CNODE_solution[(grid_B.Nu + 1):end, :, :],
    grid_B.nvx,
    grid_B.nvy,
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
u_steady = reshape(steady_CNODE_solution[1:(grid_B.Nu), :, :],
    grid_B.nux,
    grid_B.nuy,
    size(steady_CNODE_solution, 2),
    :)
v_steady = reshape(steady_CNODE_solution[(grid_B.Nu + 1):end, :, :],
    grid_B.nvx,
    grid_B.nvy,
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
    gif(anim, "plots/03.01_Burgers.gif", fps = 8)
else
    gif(anim, "examples/plots/03.01_Burgers.gif", fps = 8)
end
