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
import CoupledNODE: Laplacian, first_derivatives
using Zygote
import ("./03.01-Burgers.jl")
ν = 0.001f0
force_params = (ν,)
grid_B_dns = (grid_u_dns,)
grid_B_les = (grid_u_les,)
F_dns = create_burgers_rhs(grid_B_dns, force_params)
F_les = create_burgers_rhs(grid_B_les, force_params)

# and generate some initial conditions
u0_dns = generate_initial_conditions(grid_B_dns[1].nx, 3);

# ### Filter
using SparseArrays, Plots
# To get the LES, we use a Gaussian filter kernel, truncated to zero outside of $3 / 2$ filter widths.
ΔΦ = 5 * grid_B_les[1].dx
## Filter kernel
gaussian(Δ, x) = sqrt(6 / π) / Δ * exp(-6x^2 / Δ^2)
top_hat(Δ, x) = (abs(x) ≤ Δ / 2) / Δ
kernel = gaussian
## Discrete filter matrix (with periodic extension and threshold for sparsity)
Φ = sum(-1:1) do z
    z *= 2π
    d = @. xles - xdns' - z
    @. kernel(ΔΦ, d) * (abs(d) ≤ 3 / 2 * ΔΦ)
end
Φ = Φ ./ sum(Φ; dims = 2) ## Normalize weights
Φ = sparse(Φ)
dropzeros!(Φ)
heatmap(Φ; yflip = true, xmirror = true, title = "Filter matrix")

# Apply the filter to the initial condition
u0_les = Φ * u0_dns

# TODO: Filter should be generated from a function

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
