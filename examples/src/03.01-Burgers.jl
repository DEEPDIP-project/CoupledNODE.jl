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
nux = 300
dux = 2π / nux
grid_u = Grid(dim = 1, dx = dux, nx = nux)

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
grid_B = (grid_u,)

# Now we can create the right-hand side of the NODE
F = create_burgers_rhs(grid_B, force_params)

# We can now generate the initial conditions for the Burgers equation. We will use a combination of sine waves and noise to create the initial conditions.
function generate_initial_conditions(n_samples::Int)
    x = range(0, stop = 2π, length = nux)

    u0_list = Array{Float32, 2}(undef, nux, n_samples)

    for j in 1:n_samples
        u0 = @view u0_list[:, j]
        u0 .= 0.0

        # Randomize the number of sine waves for each sample
        n_waves = rand(1:3)
        for _ in 1:n_waves
            # Randomize the frequency, amplitude, and phase shift for each sine wave
            freq = rand() * 2π
            amplitude = rand()
            phase_shift = rand() * 2π
            u0 .+= amplitude * sin.(freq * x + phase_shift)
        end

        # Randomize the amount of noise
        noise_level = rand() * 0.2
        u0 .+= noise_level * randn(nux)
    end
    return u0_list
end

u_0 = generate_initial_conditions(4)


include("./../coupling_functions/functions_NODE.jl")
f_CNODE = create_f_CNODE(create_burgers_rhs, force_params, grid_B; is_closed = false);
import Random, LuxCUDA, Lux
rng = Random.seed!(1234)
θ, st = Lux.setup(rng, f_CNODE);


# The first phase of the Burger solution will be the formation of the shock. We use a small time step to resolve the shock formation.
import DiffEqFlux: NeuralODE
t_shock = 25.0f0
dt_shock = 0.005f0
trange_burn = (0.0f0, t_shock)
saveat_shock = 0.01f0
shock_CNODE = NeuralODE(f_CNODE,
    trange_burn,
    solver_algo,
    adaptive = false,
    dt = dt_shock,
    saveat = saveat_shock);
u_shock = Array(shock_CNODE(u_0, θ, st)[1])

# Plot 
using Plots 
x = range(0, stop = 2π, length = nux)
anim = Animation()
fig = plot(layout = (4, 1), size = (300, 700))
@gif for i in 1:2:size(u_shock, 3)
    p1 = plot(x, u_shock[:, 1, i], xlabel = "x", ylabel = "u", legend=false)
    p2 = plot(x, u_shock[:, 2, i], xlabel = "x", ylabel = "u", legend=false)
    p3 = plot(x, u_shock[:, 3, i], xlabel = "x", ylabel = "u", legend=false)
    p4 = plot(x, u_shock[:, 4, i], xlabel = "x", ylabel = "u", legend=false)
    fig = plot(p1, p2, p3, p4, layout = (4, 1))
    frame(anim, fig)
end
if isdir("./plots")
    gif(anim, "plots/03.01_Burgers.gif", fps = 12)
else
    gif(anim, "examples/plots/03.01_Burgers.gif", fps = 12)
end