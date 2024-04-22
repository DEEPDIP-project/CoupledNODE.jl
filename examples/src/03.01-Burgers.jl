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

# TODO: organize this example better (maybe split?)
# what do we see here?
# the a priori fitting works (low error), but the solver is not so good
# Do we implement here Toby's approach?
# yes but first we need to move large functions to the src folder!

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
nux_dns = 1024
dux_dns = 2π / (nux_dns + 1)
grid_u_dns = Grid(dim = 1, dx = dux_dns, nx = nux_dns)
nux_les = 32
dux_les = 2π / (nux_les + 1)
grid_u_les = Grid(dim = 1, dx = dux_les, nx = nux_les)

# The following function constructs the right-hand side of the Burgers equation:
import CoupledNODE: Laplacian, first_derivatives
using Zygote
function create_burgers_rhs_central(grids, force_params)
    ν = force_params[1]

    function Force(u)
        F = Zygote.@ignore -u .* first_derivatives(u, grids[1].dx) +
                           ν * Laplacian(u, grids[1].dx^2)
        return F
    end
    return Force
end
# However the one above is not  a good discretization for
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
#
# We can implement this as follows:
function create_burgers_rhs(grids, force_params)
    ν = force_params[1]
    Δx = grids[1].dx

    function Force(u)
        # circshift handles periodic boundary conditions
        u₊ = circshift(u, -1)
        μ₊ = @. ν + Δx * abs(u + u₊) / 4 - Δx * (u₊ - u) / 12
        ϕ₊ = @. (u^2 + u * u₊ + u₊^2) / 6 - μ₊ * (u₊ - u) / Δx
        ϕ₋ = circshift(ϕ₊, 1)
        return @. -(ϕ₊ - ϕ₋) / Δx
    end
    return Force
end

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

function generate_initial_conditions(
        nx,
        nsample;
        kmax = 16,
        decay = k -> (1 + abs(k))^(-6 / 5)
)
    ## Fourier basis
    basis = [exp(2π * im * k * x / nx) for x in 1:nx, k in (-kmax):kmax]

    ## Fourier coefficients with random phase and amplitude
    c = [randn() * exp(-2π * im * rand()) * decay(k) for k in (-kmax):kmax, _ in 1:nsample]

    ## Random data samples (real-valued)
    ## Note the matrix product for summing over $k$
    real.(basis * c)
end

u0_dns = generate_initial_conditions(grid_B_dns[1].nx, 3);
# To get the initial condition of the LES we filter the data already generated
# TODO make the x array part of the grid structure
xdns = range(0, stop = 2π, length = nux_dns + 1)
xdns = xdns[1:(end - 1)]
xles = range(0, stop = 2π, length = nux_les + 1)
xles = xles[1:(end - 1)]

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

using Plots
plot(xles, u0_les, layout = (3, 1), size = (800, 300),
    label = "LES", xlabel = "x", ylabel = "u", linetype = :steppre)
plot!(xdns, u0_dns, linetype = :steppre, label = "DNS")
# Plot with periodicity to check if continuity is correct
width = 2π
xles2 = [xles; xles .+ width]
u0_les2 = [u0_les; u0_les]
xdns2 = [xdns; xdns .+ width]
u0_dns2 = [u0_dns; u0_dns]
plot(xles2, u0_les2, layout = (3, 1), size = (800, 300),
    label = "LES", xlabel = "x", ylabel = "u", linetype = :steppre)
plot!(xdns2, u0_dns2, linetype = :steppre, label = "DNS")

# Plot the differences
plot(xles2[1:(end - 1)], diff(u0_les2, dims = 1), layout = (3, 1), size = (800, 300),
    label = "LES", xlabel = "x", ylabel = "diff", linetype = :steppre)
plot!(xdns2[1:(end - 1)], diff(u0_dns2, dims = 1), linetype = :steppre, label = "DNS")

# Create the right-hand side of the NODE
# TODO: make this the src, but make it compatible with previous examples
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
plot(xles, outf_les, layout = (3, 1), size = (800, 300),
    label = "LES", xlabel = "x", ylabel = "F", linetype = :steppre)
plot!(xdns, outf_dns, linetype = :steppre, label = "DNS")
# Plot with periodicity
xdns2 = [xdns; xdns .+ width]
outf_dns2 = [outf_dns; outf_dns]
xles2 = [xles; xles .+ width]
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
# (this fail because of the LES, but this is actually the point of the example)
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
    gif(anim, "plots/03.01_Burgers.gif", fps = 10)
else
    gif(anim, "examples/plots/03.01_Burgers.gif", fps = 10)
end

# ## A-priori fitting

# Generate data
nsamples = 50
# since there are some ill initial conditions, we generate the data in batches and concatenate them
all_u_dns = zeros(size(u_dns)[1], nsamples, size(u_dns)[3])
batch_size = 10
n_batches = Int(nsamples / batch_size)
for i in 1:n_batches
    good = 0
    all_u_dns_batch = zeros(size(u_dns)[1], batch_size, size(u_dns)[3])
    while good < size(u_dns)[3]
        println("Regenerating batch $(i) (size: $(good) < $(size(u_dns)[3]))")
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
plot(xles, target_F[:, 1, 1], label = " Filtered DNS",
    xlabel = "x", ylabel = "F", linetype = :steppre)
plot!(xdns, all_F_dns[:, 1, 1], label = "DNS", linetype = :steppre)
plot!(xles, F_les(all_u_les[:, 1, :])[:, 1], label = "LES", linetype = :steppre)
# This is what we are trying to learn
plot(xles, target_F[:, 1, 1] - F_les(all_u_les[:, 1, :])[:, 1], xlabel = "x",
    ylabel = "Commutator error", linetype = :steppre, legend = false)
i = 3
plot!(xles, target_F[:, i, 1] - F_les(all_u_les[:, i, :])[:, 1], linetype = :steppre)
i = 4
plot!(xles, target_F[:, i, 1] - F_les(all_u_les[:, i, :])[:, 1], linetype = :steppre)
i = 5
plot!(xles, target_F[:, i, 1] - F_les(all_u_les[:, i, :])[:, 1], linetype = :steppre)

# Now create the the Neural Network
#import CoupledNODE: create_fno_model
using NNlib: gelu
include("./../../src/FNO.jl")
ch_fno = [5, 5, 5, 5];
kmax_fno = [16, 16, 16, 8];
σ_fno = [gelu, gelu, gelu, identity];
NN_u = create_fno_model(kmax_fno, ch_fno, σ_fno, grid_B_les[1]);

#region
#using NNlib, ComponentArrays, FFTW
#
#function create_model(chain, rng)
#    ## Create parameter vector and empty state
#    θ, state = Lux.setup(rng, chain)
#
#    ## Convert nested named tuples of arrays to a ComponentArray,
#    ## which behaves like a long vector
#    θ = ComponentArray(θ)
#
#    ## Convenience wrapper for empty state in input and output
#    m(v, θ) = first(chain(v, θ, state))
#
#    ## Return model and initial parameters
#    m, θ
#end
#struct FourierLayer{A,F} <: Lux.AbstractExplicitLayer
#    kmax::Int
#    cin::Int
#    cout::Int
#    σ::A
#    init_weight::F
#end
#
#FourierLayer(kmax, ch::Pair{Int,Int}; σ = identity, init_weight = glorot_uniform) =
#    FourierLayer(kmax, first(ch), last(ch), σ, init_weight)
#
#length(methods(Lux.initialparameters))
#
#Lux.initialparameters(rng::AbstractRNG, (; kmax, cin, cout, init_weight)::FourierLayer) = (;
#    spatial_weight = init_weight(rng, cout, cin),
#    spectral_weights = init_weight(rng, kmax + 1, cout, cin, 2),
#)
#Lux.initialstates(::AbstractRNG, ::FourierLayer) = (;)
#Lux.parameterlength((; kmax, cin, cout)::FourierLayer) =
#    cout * cin + (kmax + 1) * 2 * cout * cin
#Lux.statelength(::FourierLayer) = 0
#
### Pretty printing
#function Base.show(io::IO, (; kmax, cin, cout, σ)::FourierLayer)
#    print(io, "FourierLayer(", kmax)
#    print(io, ", ", cin, " => ", cout)
#    print(io, "; σ = ", σ)
#    print(io, ")")
#end
#
### One more method now
#length(methods(Lux.initialparameters))
#
### This makes FourierLayers callable
#function ((; kmax, cout, cin, σ)::FourierLayer)(x, params, state)
#    nx = size(x, 1)
#
#    ## Destructure params
#    ## The real and imaginary parts of R are stored in two separate channels
#    W = params.spatial_weight
#    W = reshape(W, 1, cout, cin)
#    R = params.spectral_weights
#    R = selectdim(R, 4, 1) .+ im .* selectdim(R, 4, 2)
#
#    ## Spatial part (applied point-wise)
#    y = reshape(x, nx, 1, cin, :)
#    y = sum(W .* y; dims = 3)
#    y = reshape(y, nx, cout, :)
#
#    ## Spectral part (applied mode-wise)
#    ##
#    ## Steps:
#    ##
#    ## - go to complex-valued spectral space
#    ## - chop off high wavenumbers
#    ## - multiply with weights mode-wise
#    ## - pad with zeros to restore original shape
#    ## - go back to real valued spatial representation
#    ikeep = 1:kmax+1
#    nkeep = kmax + 1
#    z = rfft(x, 1)
#    z = z[ikeep, :, :]
#    z = reshape(z, nkeep, 1, cin, :)
#    z = sum(R .* z; dims = 3)
#    z = reshape(z, nkeep, cout, :)
#    z = vcat(z, zeros(nx ÷ 2 + 1 - kmax - 1, size(z, 2), size(z, 3)))
#    z = irfft(z, nx, 1)
#
#    ## Outer layer: Activation over combined spatial and spectral parts
#    ## Note: Even though high wavenumbers are chopped off in `z` and may
#    ## possibly not be present in the input at all, `σ` creates new high
#    ## wavenumbers. High wavenumber functions may thus be represented using a
#    ## sequence of Fourier layers. In this case, the `y`s are the only place
#    ## where information contained in high input wavenumbers survive in a
#    ## Fourier layer.
#    w = σ.(z .+ y)
#
#    ## Fourier layer does not modify state
#    w, state
#end
#
#
#function create_fno(; channels, kmax, activations, rng, input_channels = (u -> u,))
#    ## Add number of input channels
#    channels = [length(input_channels); channels]
#
#    ## Model
#    create_model(
#        Chain(
#            ## Create singleton channel
#            #u -> reshape(u, size(u, 1), 1, size(u, 2)),
#
#            ## Create input channels
#            u -> hcat(map(i -> i(u), input_channels)...),
#
#            ## Some Fourier layers
#            (
#                FourierLayer(kmax[i], channels[i] => channels[i+1]; σ = activations[i]) for
#                i ∈ eachindex(kmax)
#            )...,
#
#            ## Put channels in first dimension
#            u -> permutedims(u, (2, 1, 3)),
#
#            ## Compress with a final dense layer
#            Dense(channels[end] => 2 * channels[end], gelu),
#            Dense(2 * channels[end] => 1; use_bias = false),
#
#            ## Put channels back after spatial dimension
#            u -> permutedims(u, (2, 1, 3)),
#
#            ## Remove singleton channel
#            u -> reshape(u, size(u, 1), size(u, 3)),
#        ),
#        rng,
#    )
#end
#
## ## Getting to business: Training and comparing closure models
##
## We now create a closure model. Note that the last activation is `identity`, as we
## don't want to restrict the output values. We can inspect the structure in the
## wrapped Lux `Chain`.
#
#
#m_fno, θ_fno = create_fno(;
#    channels = [5, 5, 5, 5],
#    kmax = [16, 16, 16, 8],
#    activations = [gelu, gelu, gelu, identity],
#    #input_channels = (u -> u, u -> u .^ 2),
#    input_channels = (u -> u,),
#    rng,
#)
#m_fno.chain
#
#NN_u = m_fno.chain
#endregion

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
    p1 = plot(xdns, u_dns_test[:, 1, i], xlabel = "x", ylabel = "u",
        linetype = :steppre, label = "DNS")
    plot!(xles, u_dns_test_filtered[:, 1, i], linetype = :steppre, label = "Filtered DNS")
    plot!(xles, u_les_test[:, 1, i], linetype = :steppre, label = "LES")
    plot!(xles, u_trained_test[:, 1, i], linetype = :steppre, label = "Trained")
    p2 = plot(xdns, u_dns_test[:, 2, i], xlabel = "x", ylabel = "u",
        linetype = :steppre, legend = false)
    plot!(xles, u_dns_test_filtered[:, 2, i], linetype = :steppre, legend = false)
    plot!(xles, u_les_test[:, 2, i], linetype = :steppre, legend = false)
    plot!(xles, u_trained_test[:, 2, i], linetype = :steppre, legend = false)
    p3 = plot(xdns, u_dns_test[:, 3, i], xlabel = "x", ylabel = "u",
        linetype = :steppre, legend = false)
    plot!(xles, u_dns_test_filtered[:, 3, i], linetype = :steppre, legend = false)
    plot!(xles, u_les_test[:, 3, i], linetype = :steppre, legend = false)
    plot!(xles, u_trained_test[:, 3, i], linetype = :steppre, legend = false)
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
# First reset the NN
NN_u_pos = create_fno_model(kmax_fno, ch_fno, σ_fno, grid_B_les[1]);
NNs_pos = (NN_u_pos,);
f_CNODE_pos = create_f_CNODE(
    create_burgers_rhs, force_params, grid_B_les, NNs_pos; is_closed = true)
θ_pos, st_pos = Lux.setup(rng, f_CNODE_pos);
f_CNODE_pos(all_u_les, θ_pos, st_pos)

function create_randloss_MulDtO(
        target, training_CNODE, st; nunroll, nintervals = 1,
        noverlaps = 1, nsamples, λ_c = 1e2, λ_l1 = 1e-1)
    # TODO: there should be some check about the consistency of the input arguments
    # Get the number of time steps 
    d = ndims(target)
    nt = size(target, d)
    function randloss_MulDtO(θ)
        # Compute the requested length of consecutive timesteps
        # Notice that each interval is long nunroll+1 because we are including the initial conditions as step_0 
        length_required = nintervals * (nunroll + 1) - noverlaps * (nintervals - 1)
        # Zygote will select a random initial condition that can accomodate all the multishooting intervals
        istart = Zygote.@ignore rand(1:(nt - length_required))
        trajectory = Zygote.@ignore ArrayType(selectdim(target,
            d,
            istart:(istart + length_required)))
        # and select a certain number of samples
        trajectory = Zygote.@ignore trajectory[:, rand(1:size(trajectory, 2), nsamples), :]
        # then return the loss for each multishooting set
        loss_MulDtO_oneset(trajectory,
            θ,
            st,
            training_CNODE,
            nunroll = nunroll,
            nintervals = nintervals,
            noverlaps = noverlaps,
            nsamples = nsamples,
            λ_c = λ_c,
            λ_l1 = λ_l1)
    end
end
function loss_MulDtO_oneset(trajectory,
        θ, st,
        training_CNODE;
        λ_c = 1e1,
        λ_l1 = 1e1,
        nunroll,
        nintervals,
        noverlaps,
        nsamples = nsamples)
    # Get the timesteps where the intervals start 
    starting_points = [i == 0 ? 1 : i * (nunroll + 1 - noverlaps)
                       for i in 0:(nintervals - 1)]
    # Take all the time intervals and concatenate them in the batch dimension
    list_tr = cat([trajectory[:, :, i:(i + nunroll)]
                   for i in starting_points]...,
        dims = 2)
    # Get all the initial conditions 
    list_starts = cat([trajectory[:, :, i] for i in starting_points]...,
        dims = 2)
    # Use the differentiable solver to get the predictions
    pred, target = predict_u_CNODE(list_starts, θ, st, training_CNODE, list_tr)
    # the loss is the sum of the differences between the real trajectory and the predicted one
    loss = sum(abs2, target[:, :, 2:end] .- pred[:, :, 2:end]) ./
           sum(abs2, target[:, :, 2:end])

    if λ_c > 0 && size(target, 3) == nunroll + 1
        # //TODO check if the continuity term is correct
        # Compute the continuity term by comparing end of one interval with the start of the next one
        # (!) Remind that the trajectory is stored as: 
        #   pred[grid, (nintervals*nsamples), nunroll+1]
        # and we need to compare the last noverlaps points of an interval
        pred_end = pred[:, :, (end - noverlaps + 1):end]
        # with the first noverlaps points of the next interval EXCLUDING the initial condition 
        # (which is already part of the loss function)
        pred_start = pred[:, :, 2:(1 + noverlaps)]
        continuity = 0
        # loop over all the samples, which have been concatenated in dim 2
        for s in 1:nsamples
            # each sample contains nintervals, we need to shift the index by
            s_shift = (s - 1) * nintervals
            # loop over all the intervals for the sample (excluding the last one)
            for i in 1:(nintervals - 1)
                continuity += sum(abs,
                    pred_end[:, s_shift + i] .- pred_start[:, s_shift + i + 1])
            end
        end
    else
        continuity = 0
    end

    return loss + (continuity * λ_c) + λ_l1 * norm(θ), nothing
end
function predict_u_CNODE(uv0, θ, st, training_CNODE, tg)
    sol = Array(training_CNODE(uv0, θ, st)[1])
    return sol[:, :, 1:nunroll], tg[:, :, 1:nunroll]
end
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
    λ_c = 1e3,
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
    p1 = plot(xdns, u_dns_test[:, 1, i], xlabel = "x", ylabel = "u",
        linetype = :steppre, label = "DNS")
    plot!(xles, u_les_test[:, 1, i], linetype = :steppre, label = "LES")
    plot!(xles, u_trained_test[:, 1, i], linetype = :steppre, label = "A-priori")
    plot!(xles, u_posteriori_test[:, 1, i], linetype = :steppre, label = "A-posteriori")
    p2 = plot(xdns, u_dns_test[:, 2, i], xlabel = "x", ylabel = "u",
        linetype = :steppre, legend = false)
    plot!(xles, u_les_test[:, 2, i], linetype = :steppre, legend = false)
    plot!(xles, u_trained_test[:, 2, i], linetype = :steppre, legend = false)
    plot!(xles, u_posteriori_test[:, 2, i], linetype = :steppre, legend = false)
    p3 = plot(xdns, u_dns_test[:, 3, i], xlabel = "x", ylabel = "u",
        linetype = :steppre, legend = false)
    plot!(xles, u_les_test[:, 3, i], linetype = :steppre, legend = false)
    plot!(xles, u_trained_test[:, 3, i], linetype = :steppre, legend = false)
    plot!(xles, u_posteriori_test[:, 3, i], linetype = :steppre, legend = false)
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
