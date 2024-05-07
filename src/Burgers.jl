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

# ### Filter
using SparseArrays, Plots
# To get the LES, we use a Gaussian filter kernel, truncated to zero outside of $3 / 2$ filter widths.
function create_filter_matrix(grid_B_les, grid_B_dns, ΔΦ, kernel_type)
    ## Filter kernels
    gaussian(Δ, x) = sqrt(6 / π) / Δ * exp(-6x^2 / Δ^2)
    top_hat(Δ, x) = (abs(x) ≤ Δ / 2) / Δ

    ## Choose kernel
    kernel = kernel_type == "gaussian" ? gaussian : top_hat

    ## Discrete filter matrix (with periodic extension and threshold for sparsity)
    Φ = sum(-1:1) do z
        z *= 2π
        d = @. grid_B_les[1].x - grid_B_dns[1].x' - z
        @. kernel(ΔΦ, d) * (abs(d) ≤ 3 / 2 * ΔΦ)
    end
    Φ = Φ ./ sum(Φ; dims = 2) ## Normalize weights
    Φ = sparse(Φ)
    dropzeros!(Φ)
    return Φ
end

# # FORCE
# The following function constructs the right-hand side of the Burgers equation:
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

    function Force(u, args...)
        # circshift handles periodic boundary conditions
        u₊ = circshift(u, -1)
        μ₊ = @. ν + Δx * abs(u + u₊) / 4 - Δx * (u₊ - u) / 12
        ϕ₊ = @. (u^2 + u * u₊ + u₊^2) / 6 - μ₊ * (u₊ - u) / Δx
        ϕ₋ = circshift(ϕ₊, 1)
        return @. -(ϕ₊ - ϕ₋) / Δx
    end
    return Force
end
