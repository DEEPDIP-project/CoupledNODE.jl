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
nux_dns = 1000
dux_dns = 2π / nux_dns
grid_u_dns = Grid(dim = 1, dx = dux_dns, nx = nux_dns)
nux_les = 40 # This is I in Toby's paper
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
u0_dns = generate_initial_conditions(grid_B_dns[1].nx, 1, kmax = 4);

# Use a gaussian filter to get the coarse grid
ΔΦ = 5 * grid_B_les[1].dx
Φ = create_filter_matrix(grid_B_les, grid_B_dns, ΔΦ, "gaussian")
# or a top hat filter similar to the one used by Toby
ΔΦ = 3 * grid_B_les[1].dx
Φ = create_filter_matrix(grid_B_les, grid_B_dns, ΔΦ, "hat")
# Apply the filter to the initial condition
u0_les = Φ * u0_dns

# ### Subgrid scale (SGS) 
# The Subgrid scale (SGS) is defined as the difference between the DNS and the reconstructed LES.
# Let's show an example of the SGS term for the Burgers equation:
# To get the reconstruction operator I need the small cell volume ω and the large cell volume Ω (nha that is only for average)
ω = grid_B_dns[1].dx
Ω = grid_B_les[1].dx
R = 1 / ω * transpose(Φ) * Ω
# We construct R as a pseudo-inverse operator via the following
R = transpose(Φ) * inv(Matrix(Φ * transpose(Φ)))
# And we use it to reconstruct the LES and create the SGS
u0_rec = R * u0_les
sgs = u0_dns - u0_rec

# Let's plot a comparison of the different terms
using LaTeXStrings
plot(grid_B_dns[1].x, u0_dns, label = "u", title = "Subgrid scale (SGS)",
    xlabel = "x", ylabel = L"u", legend = :topleft)
plot!(grid_B_les[1].x, u0_les, seriestype = :stepmid, label = L"\bar{u}=\mathbf{\Phi} u")
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
# First we have to solve the dynamics
import DiffEqFlux: NeuralODE
include("./../../src/NODE.jl")
f_dns = create_f_CNODE((F_dns,), grid_B_dns; is_closed = false);
using Random, LuxCUDA, Lux
Random.seed!(123)
rng = Random.default_rng()
θ_dns, st_dns = Lux.setup(rng, f_dns);
t_sim = 40.0f0
dt_dns = 0.005f0
trange = (0.0f0, t_sim)
saveat = 0.01f0
dns = NeuralODE(f_dns,
    trange,
    solver_algo,
    adaptive = false,
    dt = dt_dns,
    saveat = saveat);
u_dns = Array(dns(u0_dns, θ_dns, st_dns)[1]);
# Drop sample dimension
u_dns = u_dns[:, 1, :]
u_filt = Φ * u_dns

E_dns = sum(u_dns .^ 2, dims = 1) * grid_B_dns[1].dx / 2
E_filt = sum(u_filt .^ 2, dims = 1) * grid_B_les[1].dx / 2

plot(E_dns[1, :], label = L"E", title = "Energy",
    xlabel = "Time steps", ylabel = "Energy")
plot!(E_filt[1, :], label = L"\bar{E}")

# ### SGS projection
# First we get the sgs
u_prime = u_dns - R * u_filt
# this is a solution living in the same dimension as the DNS, but this is too expensive to compute.
# For this reason we project it onto a lower dimensional space.
# [!] Notice that this SGS space will be coarser than the DNS, but it does NOT have to have the same dimensionality as the associated LES that we plan to solve!
# Toby would use a single value decomposition (SVD) to get the projection matrix $\bm{T}$.
# We propose instead to use principal component analysis (PCA) to get the projection matrix
using MultivariateStats, LinearAlgebra
# and we compute it using only a part of the data
ndata = size(u_prime, 2)
split_idx = Int(floor(0.9 * ndata))
permuted_idxs = randperm(ndata)
train_idxs = permuted_idxs[1:split_idx]
test_idxs = permuted_idxs[(split_idx + 1):end]
train_data = u_prime[:, train_idxs]
test_data = u_prime[:, test_idxs]
# Train the PCA
sgs_size = J = 50
T = fit(PCA, train_data; maxoutdim = sgs_size)
# this plot shows the explained variance of the PCA
plot(T.prinvars, label = "PCA explained variance")
# And test its ability to reconstruct an unseen datapoint
test_reduced = predict(T, test_data)
test_reconstructed = reconstruct(T, test_reduced)
print("Reconstruction error: ", norm(test_data - test_reconstructed))
plot(grid_B_dns[1].x, test_data[:, 1], label = "Original")
plot!(grid_B_dns[1].x, test_reconstructed[:, 1], label = "Reconstructed w PCA")

# Compare energy predicted with PCA
E_prime = sum(u_prime .^ 2, dims = 1) * grid_B_dns[1].dx / 2
u_pca = predict(T, u_prime)
E_pca = sum(u_pca .^ 2, dims = 1) * grid_B_dns[1].dx / 2
plot(E_prime, E_pca, title = "Energy SGS", legend = false,
    xlabel = L"E'", ylabel = L"\frac{1}{2}s^2")
finite_inds = isfinite.(E_prime) .& isfinite.(E_pca)
E_prime_finite = E_prime[finite_inds]
E_pca_finite = E_pca[finite_inds]
scatter(E_prime_finite, E_pca_finite, title = "Energy SGS", legend = false,
    xlabel = L"E'", ylabel = L"\frac{1}{2}s^2")
# plot the diagonal line as reference
plot!([0, maximum(E_prime_finite)], [0, maximum(E_prime_finite)], label = "y=x")

# At this point we have a matrix $\bm{T}$ that projects the SGS onto a lower dimensional space.
# In order to use the sgs, we can implement a closure model for the LES that uses the information stored in the sgs.

# ## A-priori fitting

# Generate data for the a-priori fitting
nsamples = 500
nsamples = 10
ntimes = size(u_dns)[2]
all_u_dns = zeros(size(u_dns)[1], nsamples, ntimes)
batch_size = 10
n_batches = Int(nsamples / batch_size)
for i in 1:n_batches
    good = 0
    all_u_dns_batch = zeros(size(u_dns)[1], batch_size, ntimes)
    while good < ntimes
        println("Generating batch $(i) (size: $(good) < $(ntimes)")
        all_u0_dns = generate_initial_conditions(grid_B_dns[1].nx, batch_size)
        all_u_dns_batch = Array(dns(all_u0_dns, θ_dns, st_dns)[1])
        good = size(all_u_dns_batch)[3]
    end
    all_u_dns[:, ((i - 1) * batch_size + 1):(i * batch_size), :] = all_u_dns_batch
end

# ### Data filtering 
all_F_dns = F_dns(reshape(
    all_u_dns, size(all_u_dns, 1), size(all_u_dns, 2) * size(all_u_dns, 3)));
all_F_dns = reshape(all_F_dns, size(all_u_dns));
# Reshape in and target to have sample and t  in the same dimension (makes sense in a-priori fitting)
all_u_dns_flat = reshape(all_u_dns, nux_dns, size(all_u_dns)[2] * size(all_u_dns)[3]);
all_F_dns_flat = reshape(all_F_dns, nux_dns, size(all_F_dns)[2] * size(all_F_dns)[3]);
# Filter
all_u_les_flat = Φ * all_u_dns_flat
target_F_flat = Φ * all_F_dns_flat
# Train the PCA for those data
T = fit(PCA, all_u_dns_flat; maxoutdim = sgs_size)
if size(T, 2) != sgs_size
    println("Warning: PCA did use fewer components than expected, so I will reduce the dimensionality of the SGS space to $(size(T,2))")
    sgs_size = size(T, 2)
end
# Get the sgs
target_sgs_flat = predict(T, all_u_dns_flat - R * all_u_les_flat)
# the target rhs for the sgs is $T * \frac{du'}{dt}$, where $\frac{du'}{dt} = f_{dns} - R f_{les}$
target_F_sgs_flat = predict(T, all_F_dns_flat - R * target_F_flat)
# and get them back to the original shape
all_u_les = reshape(all_u_les_flat, nux_les, size(all_u_dns)[2:end]...)
target_F = reshape(target_F_flat, nux_les, size(all_F_dns)[2:end]...);
target_sgs = reshape(target_sgs_flat, sgs_size, size(target_sgs_flat)[2:end]...);
target_F_sgs = reshape(target_F_sgs_flat, sgs_size, size(target_F_sgs_flat)[2:end]...);
# concatenate input and target
all_in = vcat(all_u_les_flat, target_sgs)
target = vcat(target_F_flat, target_F_sgs)

# Now create the the Neural Network
using NNlib: gelu
include("./../../src/FNO.jl")
ch_fno = [5, 5, 5, 5];
kmax_fno = [16, 16, 16, 8];
σ_fno = [gelu, gelu, gelu, identity];
NN_u = create_fno_model(kmax_fno, ch_fno, σ_fno, grid_B_les[1]);
NN_sgs = create_fno_model(kmax_fno, ch_fno, σ_fno, grid_B_les[1]);

# pack the NNs
NNs = (NN_u, NN_sgs);

#packe the grids assuming that the sgs is the same as the LES
dux_s = 2π / sgs_size
grid_s = Grid(dim = 1, dx = 2π / sgs_size, nx = sgs_size)
grids = (grid_u_les, grid_s)

# if it works, then the unclosed cnode and the les should have the same result

# Use it to create the cnode
include("./../../src/NODE.jl")
f_CNODE = create_f_CNODE(
    (F_les, (u, v) -> v .* 0), grids, NNs; is_closed = true)
θ, st = Lux.setup(rng, f_CNODE);

# Trigger compilation and test the force
f_CNODE(all_in, θ, st)[1]

# test this F to integrate time
import DiffEqFlux: NeuralODE
dt = 0.001f0
trange = (0.0f0, 10.0f0)
saveat_shock = 0.01f0
tr = NeuralODE(f_CNODE,
    trange,
    solver_algo,
    adaptive = false,
    dt = dt,
    saveat = saveat_shock);
u0_dns = generate_initial_conditions(grid_B_dns[1].nx, 1);
u0_les = Φ * u0_dns
s0 = predict(M, u0_dns - R * u0_les)
u_tr = Array(tr(vcat(u0_les, s0), θ, st)[1]);
using Plots
anim = Animation()
fig = plot(layout = (2, 1), size = (500, 300))
@gif for i in 1:2:size(u_tr, 3)
    u = u_tr[1:32, 1, i]
    s = u_tr[33:end, 1, i]
    p1 = plot(grid_B_les[1].x, u, xlabel = "x", ylabel = "u",
        linetype = :steppre, label = "LES")
    p2 = plot(grid_s.x, s, xlabel = "x", ylabel = "s",
        linetype = :steppre, label = "SGS")
    title = "Time: $(round((i - 1) * saveat_shock, digits = 2))"
    fig = plot(p1, p2, layout = (2, 1), title = title)
    frame(anim, fig)
end

include("./../../src/loss_priori.jl")
myloss = create_randloss_derivative(all_in,
    target,
    f_CNODE,
    st;
    n_use = 1024,
    λ = 0,
    λ_c = 0);

# To initialize the training, we need some objects to monitor the procedure, and we trigger the first compilation.
lhist = [];
## Initialize and trigger the compilation of the model
using ComponentArrays
pinit = ComponentArrays.ComponentArray(θ);
print(myloss(pinit));
## [!] Check that the loss does not get type warnings, otherwise it will be slower

# We transform the NeuralODE into an optimization problem
## Select the autodifferentiation type
import OptimizationOptimisers: Optimization
adtype = Optimization.AutoZygote();
optf = Optimization.OptimizationFunction((x, p) -> myloss(x), adtype);
optprob = Optimization.OptimizationProblem(optf, pinit);

# Select the training algorithm:
# In the previous example we have used a classic gradient method like Adam:
import OptimizationOptimisers: OptimiserChain, Adam, ClipNorm
algo = OptimiserChain(Adam(1.0e-3), ClipNorm(1));

# ### Train the CNODE
import CoupledNODE: callback
# switch to train mode to enable dropout
Lux.trainmode
result_neuralode = Optimization.solve(optprob,
    algo;
    callback = callback,
    maxiters = 300);
pinit = result_neuralode.u;
θ = pinit;
optprob = Optimization.OptimizationProblem(optf, pinit);
