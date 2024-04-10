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
# $$
# \begin{equation}
# \frac{\partial u}{\partial t} = - u \frac{\partial u}{\partial x} + \nu \frac{\partial u^2}{\partial x^2} = f(u)
# \end{equation}
# $$
# where $u(x,t)$ is the velocity of the fluid, $\nu$ is the viscosity coefficient, and $(x,y)$ and $t$ are the spatial and temporal coordinates, respectively. The equation is a non-linear partial differential equation that describes the evolution of a fluid flow in one spatial dimensions. The equation is named after Johannes Martinus Burgers, who introduced it in 1948 as a simplified model for turbulence.

# We start by defining the right-hand side of the Burgers equation. We will use the finite difference method to compute the spatial derivatives. 
# So the first step is to define the grid that we are going to use.
# We define a DNS and a LES
import CoupledNODE: Grid
nux_dns = 500
dux_dns = 2π / nux_dns
grid_u_dns = Grid(dim = 1, dx = dux_dns, nx = nux_dns)
nux_les = 100
dux_les = 2π / nux_les
grid_u_les = Grid(dim = 1, dx = dux_les, nx = nux_les)

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
grid_B_dns = (grid_u_dns,)
grid_B_les = (grid_u_les,)

# Now we can create the right-hand side of the NODE
F_dns = create_burgers_rhs(grid_B_dns, force_params)
F_les = create_burgers_rhs(grid_B_les, force_params)

# We can now generate the initial conditions for the Burgers equation. We will use a combination of sine waves to play the role of a smooth component, and a square wave to induce the shock
function generate_initial_conditions(n_samples::Int, grids)
    Nx = grids[1].nx
    x = range(0, stop = 2π, length = Nx)

    u0_list = Array{Float32, 2}(undef, Nx, n_samples)

    for j in 1:n_samples
        u0 = @view u0_list[:, j]
        u0 .= 0.0

        # Smooth component (e.g., sine wave)
        smooth_amplitude = 0.5
        smooth_freq = 10.0 * rand()
        smooth_phase_shift = rand()
        u0 .+= smooth_amplitude * sin.(smooth_freq .* x .+ smooth_phase_shift)

        # Shock component (e.g., square wave)
        shock_amplitude = 1.0
        shock_position = rand() * 2π  # Position where the shock occurs
        u0[x .>= shock_position] .+= shock_amplitude

        # Normalize the initial conditions
        u0 ./= maximum(u0)

        # Add small random noise for variability
        noise_level = 0.0005
        u0 .+= noise_level * randn(Nx)

        # Ensure periodicity by making the last point equal to the first point
        u0[end] = u0[1]
    end

    return u0_list
end

u0_dns = generate_initial_conditions(3, grid_B_dns)
# To get the initial condition of the LES we filter the data already generated
using Interpolations
xdns = range(0, stop = 2π, length = nux_dns)
xles = range(0, stop = 2π, length = nux_les)
u0_les = zeros(Float32, nux_les, size(u0_dns, 2))
for i in 1:size(u0_dns, 2)
    itp = LinearInterpolation(xdns, u0_dns[:, i], extrapolation_bc = Flat())
    u0_les[:, i] = itp[xles]
end

using Plots
plot(xles, u0_les, layout = (3, 1), size = (800, 300),
    legend = false, xlabel = "x", ylabel = "u", linetype = :steppre)
plot!(xdns, u0_dns, linetype = :steppre)

# Create the right-hand side of the NODE
include("./../coupling_functions/functions_NODE.jl")
f_dns = create_f_CNODE(create_burgers_rhs, force_params, grid_B_dns; is_closed = false);
f_les = create_f_CNODE(create_burgers_rhs, force_params, grid_B_les; is_closed = false);
import Random, LuxCUDA, Lux
rng = Random.seed!(1234)
θ_dns, st_dns = Lux.setup(rng, f_dns);
θ_les, st_les = Lux.setup(rng, f_les);

# Now solve the LES and the DNS
import DiffEqFlux: NeuralODE
t_shock = 20.0f0
dt_dns = 0.001f0
dt_les = dt_dns * 0.1f0
dt_les = dt_dns
trange_burn = (0.0f0, t_shock)
saveat_shock = 0.2f0
dns = NeuralODE(f_dns,
    trange_burn,
    solver_algo,
    adaptive = false,
    dt = dt_dns,
    saveat = saveat_shock);
les = NeuralODE(f_les,
    trange_burn,
    solver_algo,
    adaptive = false,
    dt = dt_les,
    saveat = saveat_shock);
u_dns = Array(dns(u0_dns, θ_dns, st_dns)[1]);
u_les = Array(les(u0_les, θ_les, st_les)[1]);

# Plot 
using Plots
anim = Animation()
fig = plot(layout = (3, 1), size = (800, 300))
@gif for i in 1:2:size(u_dns, 3)
    p1 = plot(xdns, u_dns[:, 1, i], xlabel = "x", ylabel = "u",
        linetype = :steppre, label = "DNS")
    plot!(xles, u_les[:, 1, i], linetype = :steppre, label = "LES")
    p2 = plot(xdns, u_dns[:, 2, i], xlabel = "x", ylabel = "u",
        linetype = :steppre, legend = false)
    plot!(xles, u_les[:, 2, i], linetype = :steppre, legend = false)
    p3 = plot(xdns, u_dns[:, 3, i], xlabel = "x", ylabel = "u",
        linetype = :steppre, legend = false)
    plot!(xles, u_les[:, 3, i], linetype = :steppre, legend = false)
    title = "Time: $(round((i - 1) * saveat_shock, digits = 2))"
    fig = plot(p1, p2, p3, layout = (3, 1), title = title)
    frame(anim, fig)
end
if isdir("./plots")
    gif(anim, "plots/03.01_Burgers.gif", fps = 5)
else
    gif(anim, "examples/plots/03.01_Burgers.gif", fps = 5)
end

# ## A-priori fitting

# Generate data
nsamples = 100
all_u0_dns = generate_initial_conditions(nsamples, grid_B_dns)
all_u0_les = zeros(Float32, nux_les, size(all_u0_dns, 2))
all_F_dns = F_dns(all_u0_dns)
target_F = zeros(Float32, nux_les, size(all_u0_dns, 2))
for i in 1:size(all_u0_dns, 2)
    itp = LinearInterpolation(xdns, all_u0_dns[:, i], extrapolation_bc = Flat())
    all_u0_les[:, i] = itp[xles]
    # The target of the a-priori fitting is the filtered DNS force
    itp = LinearInterpolation(xdns, all_F_dns[:, i], extrapolation_bc = Flat())
    target_F[:, i] = itp[xles]
end

# (you have to simulate them ^ ! not only at t=0)

# ## Energy

# PDEs like the Burgers equation conserve energy. If we discretize the Burgers equation the energy conservation takes the following form:
# $$
# \begin{equation}
# \frac{dE}{dt} = \bm{u}^T \bm{\omega} f(\bm{u}) 
# \end{equation}
# $$
# where $E$ is the energy of the system given by:
# $$
# \begin{equation}
# E = \frac{1}{2} \bm{u}^T \bm{\omega} \bm{u},
# \end{equation}
# $$
# and $\bm{\omega} \in \mathbb{R}^{N\times N}$ is the grid volumes of the diagonal elements.
# In a dissipative system as Burgers equation, the energy will decrease over time. We can compute the energy of the system at each time step and plot it to verify that the energy is decreasing.
