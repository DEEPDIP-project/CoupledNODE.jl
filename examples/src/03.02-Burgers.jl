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

# # Burgers equations with small scale

# We start by defining the right-hand side of the Burgers equation. We will use the finite difference method to compute the spatial derivatives. 
# So the first step is to define the grid that we are going to use.
# We define a DNS and a LES
import CoupledNODE: Grid
nux_dns = 1024
dux_dns = 2π / nux_dns
grid_u_dns = Grid(dim = 1, dx = dux_dns, nx = nux_dns)
nux_les = 32
dux_les = 2π / nux_les
grid_u_les = Grid(dim = 1, dx = dux_les, nx = nux_les)

# Construct the right-hand side of the Burgers equation
include("./../../src/Burgers.jl")
ν = 0.001f0
force_params = (ν,)
grid_B_dns = (grid_u_dns,)
grid_B_les = (grid_u_les,)
F_dns = create_burgers_rhs(grid_B_dns, force_params)
F_les = create_burgers_rhs(grid_B_les, force_params)

# and generate some initial conditions
u0_dns = generate_initial_conditions(grid_B_dns[1].nx, 3);

# Set the kernel size and get the gaussian filter
ΔΦ = 5 * grid_B_les[1].dx
Φ = create_filter_matrix(grid_B_les, grid_B_dns, ΔΦ, "gaussian")
# Apply the filter to the initial condition
u0_les = Φ * u0_dns
transpose(Φ)
# ### Subgrid scale (SGS) 
# The Subgrid scale (SGS) is defined as the difference between the DNS and the reconstructed LES.
# Let's show an example of the SGS term for the Burgers equation:
# To get the erconstruction operator I need the cell volume ω and the grid volume Ω
ω = 1.0 / grid_B_dns[1].dx
Ω = 2π
R = 1 / ω * transpose(Φ) * Ω
# NOT ok! fix R
u0_rec = R * u0_les
sgs = u0_dns - u0_rec
plot(grid_B_dns[1].x, u0_dns, label = "DNS", title = "Subgrid scale (SGS)",
    xlabel = "x", ylabel = "u", legend = :topleft)
plot!(grid_B_les[1].x, u0_les, label = "LES")
plot!(grid_B_dns[1].x, u0_rec, label = "Rec-LES")
plot!(grid_B_dns[1].x, sgs, label = "SGS")

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
