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
include("./../../src/grid.jl")
nux_dns = 1024
dux_dns = 2π / nux_dns
grid_u_dns = Grid(dim = 1, dx = dux_dns, nx = nux_dns)
nux_les = 32
dux_les = 2π / nux_les
grid_u_les = Grid(dim = 1, dx = dux_les, nx = nux_les)

include("./../../src/Burgers.jl")

# However a central method is not  a good discretization for
# dealing with shocks. Jameson proposes the following scheme instead:
#
# $$
# \begin{split}
# \frac{\mathrm{d} u_n}{\mathrm{d} t} & = - \frac{\phi_{n + 1 / 2} - \phi_{n - 1 / 2}}{\Delta x}, \\
# \phi_{n + 1 / 2} & = \frac{u_{n + 1}^2 + u_{n + 1} u_n + u_n^2}{6} - \mu_{n + 1 / 2} \frac{u_{n + 1} - u_n}{\Delta x}, \\
# \mu_{n + 1 / 2} & = \nu + \Delta x \left( \frac{| u_{n + 1} + u_n |}{4} - \frac{u_{n + 1} - u_n}{12} \right),
# \end{split}
# $$
#
# where $ϕ_{n + 1 / 2}$ is the numerical flux from $u_n$ to $u_{n + 1}$
# and $\mu_{n + 1 / 2}$ includes the original viscosity and a numerical viscosity.
# This prevents oscillations near shocks.

# Let's set the parameters for the Burgers equation
ν = 0.001f0
# and we pack them into a tuple for the rhs Constructor
force_params = (ν,)
# we also need to pack the grid into a tuple
grid_B_dns = (grid_u_dns,)
grid_B_les = (grid_u_les,)

# Now we can create the right-hand side of the NODE
F_dns = create_burgers_rhs(grid_B_dns, force_params)
F_les = create_burgers_rhs(grid_B_les, force_params)

# ### Initial conditions
#
# For the initial conditions, we use the following random Fourier series:
#
# $$
# u_0(x) = \mathfrak{R} \sum_{k = -k_\text{max}}^{k_\text{max}} c_k
# \mathrm{e}^{2 \pi \mathrm{i} k x},
# $$
#
# where
#
# - $\mathfrak{R}$ denotes the real part
# - $c_k = a_k d_k \mathrm{e}^{- 2 \pi \mathrm{i} b_k}$ are random
#   Fourier series coefficients
# - $a_k \sim \mathcal{N}(0, 1)$ is a normally distributed random amplitude
# - $d_k = (1 + | k |)^{- 6 / 5}$ is a deterministic spectral decay profile,
#   so that the large scale features dominate the initial flow
# - $b_k \sim \mathcal{U}(0, 1)$ is a uniform random phase shift between 0 and 1
# - $\mathrm{e}^{2 \pi \mathrm{i} k x}$ is a sinusoidal Fourier series basis
#   function evaluated at the point $x \in \Omega$
#
# Note in particular that the constant coefficient $c_0$ ($k = 0$) is almost
# certainly non-zero, and with complex amplitude $| c_0 | = | a_0 |$.
#
# Since the same Fourier basis can be reused multiple times, we write a
# function that creates multiple initial condition samples in one go. Each
# discrete $u_0$ vector is stored as a column in the resulting matrix.
u0_dns = generate_initial_conditions(grid_B_dns[1].nx, 3);

# ### Filter
using SparseArrays, Plots
# To get the LES, we use a Gaussian filter kernel, truncated to zero outside of $3 / 2$ filter widths.
# Usage:
ΔΦ = 5 * grid_B_les[1].dx
Φ = create_filter_matrix(grid_B_les, grid_B_dns, ΔΦ, "gaussian")
heatmap(Φ; yflip = true, xmirror = true, title = "Filter matrix")

# Apply the filter to the initial condition
u0_les = Φ * u0_dns

using Plots
plot(grid_B_les[1].x, u0_les, layout = (3, 1), size = (800, 300),
    label = "LES", xlabel = "x", ylabel = "u", linetype = :steppre)
plot!(grid_B_dns[1].x, u0_dns, linetype = :steppre, label = "DNS")
# Plot with periodicity to check if continuity is correct
width = 2π
xles2 = [grid_B_les[1].x; grid_B_les[1].x .+ width]
u0_les2 = [u0_les; u0_les]
xdns2 = [grid_B_dns[1].x; grid_B_dns[1].x .+ width]
u0_dns2 = [u0_dns; u0_dns]
plot(xles2, u0_les2, layout = (3, 1), size = (800, 300),
    label = "LES", xlabel = "x", ylabel = "u", linetype = :steppre)
plot!(xdns2, u0_dns2, linetype = :steppre, label = "DNS")

# Plot the differences
plot(xles2[1:(end - 1)], diff(u0_les2, dims = 1), layout = (3, 1), size = (800, 300),
    label = "LES", xlabel = "x", ylabel = "diff", linetype = :steppre)
plot!(xdns2[1:(end - 1)], diff(u0_dns2, dims = 1), linetype = :steppre, label = "DNS")

# Create the right-hand side of the NODE
# TODO: is this compatible with previous examples?
include("./../../src/NODE.jl")
f_dns = create_f_CNODE(create_burgers_rhs, force_params, grid_B_dns; is_closed = false);
f_les = create_f_CNODE(create_burgers_rhs, force_params, grid_B_les; is_closed = false);
using Random, LuxCUDA, Lux
Random.seed!(123)
rng = Random.default_rng()
θ_dns, st_dns = Lux.setup(rng, f_dns);
θ_les, st_les = Lux.setup(rng, f_les);

# Plot the forces
outf_dns = Array(f_dns(u0_dns, θ_dns, st_dns)[1])
outf_les = Array(f_les(u0_les, θ_les, st_les)[1])
plot(grid_B_les[1].x, outf_les, layout = (3, 1), size = (800, 300),
    label = "LES", xlabel = "x", ylabel = "F", linetype = :steppre)
plot!(grid_B_dns[1].x, outf_dns, linetype = :steppre, label = "DNS")
# Plot with periodicity
outf_dns2 = [outf_dns; outf_dns]
outf_les2 = [outf_les; outf_les]
plot(xles2, outf_les2, layout = (3, 1), size = (800, 300),
    label = "LES", xlabel = "x", ylabel = "F", linetype = :steppre)
plot!(xdns2, outf_dns2, linetype = :steppre, label = "DNS")

# Now solve the LES and the DNS
import DiffEqFlux: NeuralODE
t_shock = 10.0f0
dt_dns = 0.001f0
dt_les = dt_dns
trange_burn = (0.0f0, t_shock)
saveat_shock = 0.01f0
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
    p1 = plot(grid_B_dns[1].x, u_dns[:, 1, i], xlabel = "x", ylabel = "u",
        linetype = :steppre, label = "DNS")
    plot!(grid_B_les[1].x, u_les[:, 1, i], linetype = :steppre, label = "LES")
    p2 = plot(grid_B_dns[1].x, u_dns[:, 2, i], xlabel = "x", ylabel = "u",
        linetype = :steppre, legend = false)
    plot!(grid_B_les[1].x, u_les[:, 2, i], linetype = :steppre, legend = false)
    p3 = plot(grid_B_dns[1].x, u_dns[:, 3, i], xlabel = "x", ylabel = "u",
        linetype = :steppre, legend = false)
    plot!(grid_B_les[1].x, u_les[:, 3, i], linetype = :steppre, legend = false)
    title = "Time: $(round((i - 1) * saveat_shock, digits = 2))"
    fig = plot(p1, p2, p3, layout = (3, 1), title = title)
    frame(anim, fig)
end
if isdir("./plots")
    gif(anim, "plots/03.01_Burgers.gif", fps = 10)
else
    gif(anim, "examples/plots/03.01_Burgers.gif", fps = 10)
end

# ## A-priori fitting

# Generate data
nsamples = 500
nsamples = 50
# since there are some ill initial conditions, we generate the data in batches and concatenate them
all_u_dns = zeros(size(u_dns)[1], nsamples, size(u_dns)[3])
batch_size = 10
n_batches = Int(nsamples / batch_size)
for i in 1:n_batches
    good = 0
    all_u_dns_batch = zeros(size(u_dns)[1], batch_size, size(u_dns)[3])
    while good < size(u_dns)[3]
        println("Generating batch $(i) (size: $(good) < $(size(u_dns)[3]))")
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
# and get them back to the original shape
all_u_les = reshape(all_u_les_flat, nux_les, size(all_u_dns)[2:end]...)
target_F = reshape(target_F_flat, nux_les, size(all_F_dns)[2:end]...);

# Compare LES force vs interpolated DNS force
plot(grid_B_les[1].x, target_F[:, 1, 1], label = " Filtered DNS",
    xlabel = "x", ylabel = "F", linetype = :steppre)
plot!(grid_B_dns[1].x, all_F_dns[:, 1, 1], label = "DNS", linetype = :steppre)
plot!(grid_B_les[1].x, F_les(all_u_les[:, 1, :])[:, 1], label = "LES", linetype = :steppre)
# This is what we are trying to learn
plot(grid_B_les[1].x, target_F[:, 1, 1] - F_les(all_u_les[:, 1, :])[:, 1], xlabel = "x",
    ylabel = "Commutator error", linetype = :steppre, legend = false)
i = 3
plot!(grid_B_les[1].x, target_F[:, i, 1] - F_les(all_u_les[:, i, :])[:, 1],
    linetype = :steppre)
i = 4
plot!(grid_B_les[1].x, target_F[:, i, 1] - F_les(all_u_les[:, i, :])[:, 1],
    linetype = :steppre)
i = 5
plot!(grid_B_les[1].x, target_F[:, i, 1] - F_les(all_u_les[:, i, :])[:, 1],
    linetype = :steppre)

# Now create the the Neural Network
#import CoupledNODE: create_fno_model
using NNlib: gelu
include("./../../src/FNO.jl")
ch_fno = [5, 5, 5, 5];
kmax_fno = [16, 16, 16, 8];
σ_fno = [gelu, gelu, gelu, identity];
NN_u = create_fno_model(kmax_fno, ch_fno, σ_fno, grid_B_les[1]);

# pack the NNs
NNs = (NN_u,);

# Use it to create the cnode
include("./../../src/NODE.jl")
f_CNODE = create_f_CNODE(
    create_burgers_rhs, force_params, grid_B_les, NNs; is_closed = true)
θ, st = Lux.setup(rng, f_CNODE);

# Trigger compilation and test the force
f_CNODE(all_u_les, θ, st)

# a priori fitting
include("./../../src/loss_priori.jl")
myloss = create_randloss_derivative(all_u_les_flat,
    target_F_flat,
    f_CNODE,
    st;
    nuse = 1024,
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
    maxiters = 1000);
pinit = result_neuralode.u;
θ = pinit;
optprob = Optimization.OptimizationProblem(optf, pinit);
# (Notice that the block above can be repeated to continue training)

# Compute the error in estimating the force
error_les = sum(abs, f_les(all_u_les_flat, θ_les, st_les)[1] - target_F_flat) /
            sum(abs, target_F_flat)
error_trained_les = sum(abs, f_CNODE(all_u_les_flat, θ, st)[1] - target_F_flat) /
                    sum(abs, target_F_flat)
bar(["LES", "Trained LES"], [error_les, error_trained_les],
    title = "Comparison of errors in estimating the force",
    xlabel = "Method",
    ylabel = "Error %",
    legend = false)
# From the plot it looks like the trained LES is better than the standard LES!
# However, if we use the trained model to run a new simulation, things may not be soo good:
Lux.testmode
trained_les = NeuralODE(f_CNODE,
    trange_burn,
    solver_algo,
    adaptive = false,
    dt = dt_les,
    saveat = saveat_shock);
# Repeat this until not instble
u_dns_test = similar(u_dns) * 0;
u_les_test = similar(u_les) * 0;
u_trained_test = similar(u_les) * 0;
# generate M new samples
M = 3
println("Generating")
u0_test = generate_initial_conditions(grid_B_dns[1].nx, 10);
# test the dns 
u_dns_test = Array(dns(u0_test, θ_dns, st_dns)[1]);
# test the les
u0_test_les = Φ * u0_test
u_les_test = Array(les(u0_test_les, θ_les, st_les)[1]);
# and test the trained model
u_trained_test = Array(trained_les(u0_test_les, θ, st)[1])
# Filter the DNS data
u_dns_test_filtered = Φ * reshape(
    u_dns_test, nux_dns, size(u_dns_test)[2] * size(u_dns_test)[3]);
u_dns_test_filtered = reshape(
    u_dns_test_filtered, nux_les, size(u_dns_test)[2], size(u_dns_test)[3]);

# Plot and compare the solutions
anim = Animation()
fig = plot(layout = (3, 1), size = (800, 300))
@gif for i in 1:2:size(u_trained_test, 3)
    p1 = plot(grid_B_dns[1].x, u_dns_test[:, 1, i], xlabel = "x", ylabel = "u",
        linetype = :steppre, label = "DNS")
    plot!(grid_B_les[1].x, u_dns_test_filtered[:, 1, i],
        linetype = :steppre, label = "Filtered DNS")
    plot!(grid_B_les[1].x, u_les_test[:, 1, i], linetype = :steppre, label = "LES")
    plot!(grid_B_les[1].x, u_trained_test[:, 1, i], linetype = :steppre, label = "Trained")
    p2 = plot(grid_B_dns[1].x, u_dns_test[:, 2, i], xlabel = "x", ylabel = "u",
        linetype = :steppre, legend = false)
    plot!(
        grid_B_les[1].x, u_dns_test_filtered[:, 2, i], linetype = :steppre, legend = false)
    plot!(grid_B_les[1].x, u_les_test[:, 2, i], linetype = :steppre, legend = false)
    plot!(grid_B_les[1].x, u_trained_test[:, 2, i], linetype = :steppre, legend = false)
    p3 = plot(grid_B_dns[1].x, u_dns_test[:, 3, i], xlabel = "x", ylabel = "u",
        linetype = :steppre, legend = false)
    plot!(
        grid_B_les[1].x, u_dns_test_filtered[:, 3, i], linetype = :steppre, legend = false)
    plot!(grid_B_les[1].x, u_les_test[:, 3, i], linetype = :steppre, legend = false)
    plot!(grid_B_les[1].x, u_trained_test[:, 3, i], linetype = :steppre, legend = false)
    title = "Time: $(round((i - 1) * saveat_shock, digits = 2))"
    fig = plot(p1, p2, p3, layout = (3, 1), title = title)
    frame(anim, fig)
end
if isdir("./plots")
    gif(anim, "plots/03.01_Burgers.gif", fps = 15)
else
    gif(anim, "examples/plots/03.01_Burgers.gif", fps = 15)
end

# As you can see from the plot, the trained model produces a solution that is not stable, and over time it diverges from the DNS.
# Let's try to fix this with a posteriori fitting:

# ### A-posteriori fitting
include("./../../src/loss_posteriori.jl")
# First reset the NN
NN_u_pos = create_fno_model(kmax_fno, ch_fno, σ_fno, grid_B_les[1]);
NNs_pos = (NN_u_pos,);
f_CNODE_pos = create_f_CNODE(
    create_burgers_rhs, force_params, grid_B_les, NNs_pos; is_closed = true)
θ_pos, st_pos = Lux.setup(rng, f_CNODE_pos);
f_CNODE_pos(all_u_les, θ_pos, st_pos)

nunroll = 20
nintervals = 5
noverlaps = 1
nsamples = 3;
dt_train = dt_les;
saveat_train = saveat_shock
t_train_range = (0.0, saveat_train * (nunroll - 1))
training_CNODE = NeuralODE(f_CNODE_pos,
    t_train_range,
    Tsit5(),
    adaptive = false,
    dt = dt_train,
    saveat = saveat_train);
myloss = create_randloss_MulDtO(all_u_les,
    training_CNODE,
    st_pos,
    nunroll = nunroll,
    noverlaps = noverlaps,
    nintervals = nintervals,
    nsamples = nsamples,
    λ_c = 0, ## TODO: TEST THIS!
    λ_l1 = 0);

lhist = [];
pinit = ComponentArrays.ComponentArray(θ_pos);
print(myloss(pinit));
adtype = Optimization.AutoZygote();
optf = Optimization.OptimizationFunction((x, p) -> myloss(x), adtype);
optprob = Optimization.OptimizationProblem(optf, pinit);
import OptimizationOptimisers: OptimiserChain, Adam, ClipNorm
algo = OptimiserChain(Adam(1.0e-3), ClipNorm(1));
Lux.trainmode
# TODO: callback should be resettable
result_neuralode = Optimization.solve(optprob,
    algo;
    callback = callback,
    maxiters = 50);
pinit = result_neuralode.u;
θ_pos = pinit;
optprob = Optimization.OptimizationProblem(optf, pinit);

# Compute the error in estimating the force
error_posteriori = sum(abs, f_CNODE_pos(all_u_les_flat, θ_pos, st_pos)[1] - target_F_flat) /
                   sum(abs, target_F_flat)
bar(["LES", "A-priori fitting", "A-posteriori fitting"],
    [error_les, error_trained_les, error_posteriori],
    title = "Comparison of errors in estimating the force",
    xlabel = "Method",
    ylabel = "Error %",
    legend = false)

# and test the trained model
u_posteriori_test = Array(trained_les(u0_test_les, θ_pos, st_pos)[1])

# Plot
anim = Animation()
fig = plot(layout = (3, 1), size = (800, 300))
@gif for i in 1:2:size(u_trained_test, 3)
    p1 = plot(grid_B_dns[1].x, u_dns_test[:, 1, i], xlabel = "x", ylabel = "u",
        linetype = :steppre, label = "DNS")
    plot!(grid_B_les[1].x, u_les_test[:, 1, i], linetype = :steppre, label = "LES")
    #plot!(grid_B_les[1].x, u_trained_test[:, 1, i], linetype = :steppre, label = "A-priori")
    plot!(grid_B_les[1].x, u_posteriori_test[:, 1, i],
        linetype = :steppre, label = "A-posteriori")
    p2 = plot(grid_B_dns[1].x, u_dns_test[:, 2, i], xlabel = "x", ylabel = "u",
        linetype = :steppre, legend = false)
    plot!(grid_B_les[1].x, u_les_test[:, 2, i], linetype = :steppre, legend = false)
    #plot!(grid_B_les[1].x, u_trained_test[:, 2, i], linetype = :steppre, legend = false)
    plot!(grid_B_les[1].x, u_posteriori_test[:, 2, i], linetype = :steppre, legend = false)
    p3 = plot(grid_B_dns[1].x, u_dns_test[:, 3, i], xlabel = "x", ylabel = "u",
        linetype = :steppre, legend = false)
    plot!(grid_B_les[1].x, u_les_test[:, 3, i], linetype = :steppre, legend = false)
    #plot!(grid_B_les[1].x, u_trained_test[:, 3, i], linetype = :steppre, legend = false)
    plot!(grid_B_les[1].x, u_posteriori_test[:, 3, i], linetype = :steppre, legend = false)
    title = "Time: $(round((i - 1) * saveat_shock, digits = 2))"
    fig = plot(p1, p2, p3, layout = (3, 1), title = title)
    frame(anim, fig)
end
if isdir("./plots")
    gif(anim, "plots/03.01_Burgers.gif", fps = 15)
else
    gif(anim, "examples/plots/03.01_Burgers.gif", fps = 15)
end

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
