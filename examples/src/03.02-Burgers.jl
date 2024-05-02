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
u0_dns = generate_initial_conditions(grid_B_dns[1].nx, 1);

# Set the kernel size and get the gaussian filter
ΔΦ = 5 * grid_B_les[1].dx
Φ = create_filter_matrix(grid_B_les, grid_B_dns, ΔΦ, "hat")
Φ = create_filter_matrix(grid_B_les, grid_B_dns, ΔΦ, "gaussian")
# Apply the filter to the initial condition
u0_les = Φ * u0_dns

# ### Subgrid scale (SGS) 
# The Subgrid scale (SGS) is defined as the difference between the DNS and the reconstructed LES.
# Let's show an example of the SGS term for the Burgers equation:
# To get the reconstruction operator I need the small cell volume ω and the large cell volume Ω (nha that is only for average)
ω = grid_B_dns[1].dx
Ω = grid_B_les[1].dx
R = 1 / ω * transpose(Φ) * Ω
# Actually this is the correct way to construct the R operator
R = transpose(Φ) * inv(Matrix(Φ * transpose(Φ)))
# is this the identity?
using LinearAlgebra
isapprox(R * Φ, Matrix(I, size(R * Φ)), atol = 1e-5)
isapprox(Φ * R, Matrix(I, size(Φ * R)), atol = 1e-5)
heatmap(R)
heatmap(Φ)
heatmap(Φ * R)
# NOT ok! fix R
u0_rec = R * u0_les
sgs = u0_dns - u0_rec
using LaTeXStrings
plot(grid_B_dns[1].x, u0_dns, label = "u", title = "Subgrid scale (SGS)",
    xlabel = "x", ylabel = L"u", legend = :topleft)
plot!(grid_B_les[1].x, u0_les, label = L"\bar{u}=\mathbf{\Phi} u")
plot!(grid_B_dns[1].x, u0_rec, label = L"\mathbf{R} \bar{u}")
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
# In a dissipative system as Burgers equation, the energy will decrease over time, so the condition becomes actually
# $$
# \begin{equation}
# \frac{dE}{dt} = \bm{u}^T \bm{\omega} f(\bm{u}) \le 0.
# \end{equation}
# $$

# If we define our filtering operation to return the following sgs:
# $$
# \bm{u}' := \bm{u} - \bm{R} \bar{\bm{u}},
# $$ 
# then, the filtering transform the energy constraint as follows:
# $$
# \begin{equation}
# \frac{dE}{dt} = \bar{\bm{u}}^T \bm{\Omega} \frac{d\bar{\bm{u}}}{dt} + \left( \bm{u}'\right)^T \bm{\omega} \frac{d\bm{u}'}{dt} \le 0,
# \end{equation}
# $$
# where the energy is now decomposed as 
# $$
# \begin{align}
# E &=  \frac{1}{2} \bar{\bm{u}}^T \bm{\omega} \bar{\bm{u}} +\frac{1}{2} \left(\bm{u}'\right)^T \bm{\omega} \bm{u}'\\
# &:= \bar{E} + E',
# \end{align}
# $$
# which are the resovled and the sgs energy terms, respectively.

# However, we do not want to handle the sgs term explicitly, because it lives on the fine grid. So instead we compress it using a linear filter $\bm{T} \in \mathbb{R}^{M \times N}$ introducing 
# $$
# \bm{s} = \bm{T} \bm{u}',
# $$
# which now represents the sgs as $\bm{s} \in \mathbb{R}^{M}$.

# Then the energy conservation becomes
# $$
# \begin{equation}
# \frac{dE}{dt} = \bar{\bm{u}}^T \bm{\Omega} \frac{d\bar{\bm{u}}}{dt} +  \bm{s}^T \bm{\Omega} \frac{d\bm{s}}{dt} \le 0,
# \end{equation}
# $$
# where 
# $$
# \begin{equation}
# \frac{d\bm{s}}{dt} = \bm{T} \frac{d\bm{u}'}{dt}.
# \end{equation}
# $$

# ### Plot the energy
import DiffEqFlux: NeuralODE
include("./../../src/NODE.jl")
f_dns = create_f_CNODE(create_burgers_rhs, force_params, grid_B_dns; is_closed = false);
using Random, LuxCUDA, Lux
Random.seed!(123)
rng = Random.default_rng()
θ_dns, st_dns = Lux.setup(rng, f_dns);
t_shock = 10.0f0
dt_dns = 0.001f0
trange_burn = (0.0f0, t_shock)
saveat_shock = 0.01f0
dns = NeuralODE(f_dns,
    trange_burn,
    solver_algo,
    adaptive = false,
    dt = dt_dns,
    saveat = saveat_shock);
u_dns = Array(dns(u0_dns, θ_dns, st_dns)[1]);
# Drop sample dimension
u_dns = u_dns[:, 1, :]
u_filt = Φ * u_dns

E_dns = sum(u_dns .^ 2, dims = 1) * grid_B_dns[1].dx / 2
E_filt = sum(u_filt .^ 2, dims = 1) * grid_B_les[1].dx / 2

plot(E_dns[1, :], label = L"E", title = "Energy",
    xlabel = "Time", ylabel = "Energy")
plot!(E_filt[1, :], label = L"\bar{E}")

# Get the sgs
u_prime = u_dns - R * u_filt
# and do svd 
F = svd(u_prime)
# [...Toby does this...]
# Is it worth it compared to PCA?

# or PCA
using MultivariateStats
# as a test I get the PCA of a random half of the timesteps
ndata = size(u_prime, 2)
split_idx = Int(floor(0.9 * ndata))
permuted_idxs = randperm(ndata)
train_idxs = permuted_idxs[1:split_idx]
test_idxs = permuted_idxs[(split_idx + 1):end]
train_data = u_prime[:, train_idxs]
test_data = u_prime[:, test_idxs]
# Fit PCA
M = fit(PCA, train_data; maxoutdim = 32)
plot(M.prinvars, label = "PCA explained variance")
hline!([0.05])
# And test its ability to reconstruct an unseen datapoint
test_reduced = predict(M, test_data)
test_reconstructed = reconstruct(M, test_reduced)
print("Reconstruction error: ", norm(test_data - test_reconstructed))
plot(grid_B_dns[1].x, test_data[:, 1], label = "Original")
plot!(grid_B_dns[1].x, test_reconstructed[:, 1], label = "Reconstructed")

# Compare energy predicted with PCA
E_prime = sum(u_prime .^ 2, dims = 1) * grid_B_dns[1].dx / 2
u_pca = predict(M, u_prime)
E_pca = sum(u_pca .^ 2, dims = 1) * grid_B_dns[1].dx / 2
plot(E_prime, E_pca, title = "Energy SGS", legend = false,
    xlabel = L"E'", ylabel = L"\frac{1}{2}s^2")

finite_inds = isfinite.(E_prime) .& isfinite.(E_pca)
E_prime_finite = E_prime[finite_inds]
E_pca_finite = E_pca[finite_inds]

scatter(E_prime_finite, E_pca_finite, title = "Energy SGS", legend = false,
    xlabel = L"E'", ylabel = L"\frac{1}{2}s^2")

# Other tests to look at the SGS ???

# Do the LES + sgs
# to train the sgs part i will do the pca of u' computed exactly
