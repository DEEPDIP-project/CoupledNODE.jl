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
# In this example, we will solve the Burgers equation in 2D using the Neural ODEs framework. The Burgers equation is a fundamental equation in fluid dynamics and is given by:
# \begin{equation}
# \frac{\partial u}{\partial t} = - u \frac{\partial u}{\partial x} - v \frac{\partial u}{\partial y} + \nu \Delta u
# \frac{\partial v}{\partial t} = - u \frac{\partial v}{\partial x} - v \frac{\partial v}{\partial y} + \nu \Delta v
# \end{equation}
# where $\bm{u} = \left\{u(x,y,t), v(x,y,t)\right\}$ is the velocity field, $\nu$ is the viscosity coefficient, and $(x,y)$ and $t$ are the spatial and temporal coordinates, respectively. The equation is a non-linear partial differential equation that describes the evolution of a fluid flow in two spatial dimensions. The equation is named after Johannes Martinus Burgers, who introduced it in 1948 as a simplified model for turbulence.

# We start by defining the right-hand side of the Burgers equation. We will use the finite difference method to compute the spatial derivatives. 
# So the first step is to define the grid that we are going to use
import CoupledNODE: Grid
dux = duy = dvx = dvy = 2π / 100
nux = nuy = nvx = nvy = 100
grid_B = Grid(dux, duy, nux, nuy, dvx, dvy, nvx, nvy, convert_to_float32 = true);

# The following function constructs the right-hand side of the Burgers equation:
import CoupledNODE: Laplacian, first_derivatives
using Zygote
function create_burgers_rhs(grid, force_params)
    ν = force_params[1]

    function FG(u, v, grid)
        du_dx, du_dy = first_derivatives(u, grid.dx, grid.dy)
        dv_dx, dv_dy = first_derivatives(v, grid.dx, grid.dy)
        F = Zygote.@ignore -u .* du_dx - v .* du_dy .+ ν * Laplacian(u)
        G = Zygote.@ignore -u .* dv_dx - v .* dv_dy .+ ν * Laplacian(v)
        return F, G
    end
    return FG
end
# Notice that compared to the Gray-Scott example we are returning a single function that computes both components of the force at the same time. This is because the Burgers equation is a system of two coupled PDEs so we want to avoid recomputing the derivatives a second time.

# Let's set the parameters for the Burgers equation
ν = 0.0005f0
# and we pack them into a tuple for the rhs Constructor
force_params = (ν,)

# Now we can create the right-hand side of the NODE
FG = create_burgers_rhs(grid_B, force_params)
include("./../coupling_functions/functions_NODE.jl")
f_CNODE = create_f_CNODE(create_burgers_rhs, force_params, grid_B; is_closed = false);
import Random, LuxCUDA, Lux
rng = Random.seed!(1234)
θ, st = Lux.setup(rng, f_CNODE);

# Now we create the initial condition for the Burgers equation. For this we will need some auxiliary functions (those function have been developed by Toby to handle 2 and 3 dimensions [TODO: test])
function construct_k(grid)
    # Get the number of dimensions
    dims = grid.Nd
    # Calculate the Fourier frequencies for each dimension
    k = [fftfreq(i, i) for i in (grid.Nu, grid.Nv)]
    # Create an array of ones with the same dimensions as the input
    some_ones = ones(grid.Nu, grid.Nv)
    # Initialize k_mats with the Fourier frequencies for the first dimension
    k_mats = some_ones .* k[1]
    k_mats = reshape(k_mats, (size(k_mats)..., 1))
    # Loop over the remaining dimensions
    for i in 2:dims
        # Create a permutation of the dimensions where the current dimension is first
        original_dims = collect(1:dims)
        permuted_dims = copy(original_dims)
        permuted_dims[1], permuted_dims[i] = permuted_dims[i], permuted_dims[1]
        # Calculate the Fourier frequencies for the current dimension
        k_mat = permutedims(k[i] .* permutedims(some_ones, permuted_dims), permuted_dims)
        # Concatenate the Fourier frequencies for the current dimension to k_mats
        k_mats = cat(k_mats, k_mat, dims = dims + 1)
    end
    # Return the multi-dimensional array of Fourier frequencies
    return k_mats
end

function gen_permutations(N)
    # Get the number of dimensions
    dims = length(N)
    # Create a grid for each dimension
    N_grid = [collect(1:n) for n in N]
    # Initialize an array of ones with the same dimensions as N
    sub_grid = ones(Int, N...)
    # Preallocate sub_grids with the final size
    sub_grids = Array{Int, dims + 1}(undef, (N..., dims))
    # Loop over the dimensions
    for i in 1:dims
        # Create a permutation of the dimensions where the current dimension is first
        permuted_dims = circshift(collect(1:dims), i - 1)
        # Calculate the permuted grid for the current dimension
        permuted_grid = permutedims(
            N_grid[i] .* permutedims(sub_grid, permuted_dims), permuted_dims)
        # Assign the permuted grid to the corresponding slice of sub_grids
        sub_grids[:, :, i] = permuted_grid
    end
    # Return the reshaped sub_grids
    return reshape(sub_grids, (prod(N)..., dims))
end

function construct_spectral_filter(k_mats, max_k)
    # Get the dimensions of k_mats without the last dimension
    N = size(k_mats)[1:(end - 1)]
    # Initialize the filter as an array of ones with the same dimensions as N
    filter = ones(N)
    # Calculate the square of the maximum frequency
    max_k_squared = max_k^2
    # Loop over all permutations of the dimensions
    for i in CartesianIndices(N)
        # Calculate the square of the Euclidean norm of the Fourier frequencies for the current permutation
        k_squared = sum(k_mats[i] .^ 2)
        # If the square of the Euclidean norm is greater than or equal to the square of the maximum frequency, set the corresponding element of the filter to 0
        if k_squared >= max_k_squared
            filter[i] = 0
        end
    end
    # Return the filter
    return filter
end

function generate_random_field(grid, max_k; norm = 1, samples = (1, 1))
    # Get the number of dimensions
    dims = grid.N
    # Construct the Fourier frequencies and the spectral filter
    k = construct_k(grid)
    filter = construct_spectral_filter(k, max_k)
    # Generate random coefficients in Fourier space
    coefs = (rand(Uniform(-1, 1), (dims..., samples...)) +
             rand(Uniform(-1, 1), (dims..., samples...)) * (0 + 1im))
    # Apply the spectral filter and transform back to real space
    result = real(ifft(filter .* coefs, dims))
    # Normalize the result
    energy = sum(result .^ 2) / prod(dims)
    result = result / sqrt(energy) * norm
    return result
end

# then we can generate the initial conditions
max_k = 10
energy_norm = 1
number_of_simulations = 5
uv0 = generate_random_field(grid_B,
    max_k,
    norm = energy_norm,
    samples = (1, number_of_simulations)
)

# Short *burnout run* to get rid of the initial artifacts
trange_burn = (0.0f0, 10.0f0)
dt, saveat = (1e-2, 5)
burnout_CNODE = NeuralODE(f_CNODE,
    trange_burn,
    solver_algo,
    adaptive = false,
    dt = dt,
    saveat = saveat);
burnout_CNODE_solution = Array(burnout_CNODE(uv0, θ, st)[1]);

# We want to solve the convection diffusion equation

# this is the equation we want to solve

# write it as a system of first order ODEs

# create the grid
x = collect(LinRange(-pi, pi, 101))
y = collect(LinRange(-pi, pi, 101))
# so we get this dx and dy (constant grid)
dx = x[2] - x[1]
dy = y[2] - y[1]
# and the initial condition is random 
c0 = rand(101, 101)
# which is a scalar field because we are looking for the concentration of a single species

# the user specifies this equation
function gen_conv_diff_f(speed, viscosity, dx, dy)
    # Derivatives using finite differences
    function first_derivative(u, Δx, Δy)
        du_dx = zeros(size(u))
        du_dy = zeros(size(u))

        du_dx[:, 2:(end - 1)] = (u[:, 3:end] - u[:, 1:(end - 2)]) / (2 * Δx)
        du_dx[:, 1] = (u[:, 2] - u[:, end]) / (2 * Δx)
        du_dx[:, end] = (u[:, 1] - u[:, end - 1]) / (2 * Δx)

        du_dy[2:(end - 1), :] = (u[3:end, :] - u[1:(end - 2), :]) / (2 * Δy)
        du_dy[1, :] = (u[2, :] - u[end, :]) / (2 * Δy)
        du_dy[end, :] = (u[1, :] - u[end - 1, :]) / (2 * Δy)

        return du_dx, du_dy
    end
    function second_derivative(u, Δx, Δy)
        d2u_dx2 = zeros(size(u))
        d2u_dy2 = zeros(size(u))

        d2u_dx2[:, 2:(end - 1)] = (u[:, 3:end] - 2 * u[:, 2:(end - 1)] +
                                   u[:, 1:(end - 2)]) / (Δx^2)
        d2u_dx2[:, 1] = (u[:, 2] - 2 * u[:, 1] + u[:, end]) / (Δx^2)
        d2u_dx2[:, end] = (u[:, 1] - 2 * u[:, end] + u[:, end - 1]) / (Δx^2)

        d2u_dy2[2:(end - 1), :] = (u[3:end, :] - 2 * u[2:(end - 1), :] +
                                   u[1:(end - 2), :]) / (Δy^2)
        d2u_dy2[1, :] = (u[2, :] - 2 * u[1, :] + u[end, :]) / (Δy^2)
        d2u_dy2[end, :] = (u[1, :] - 2 * u[end, :] + u[end - 1, :]) / (Δy^2)

        return d2u_dx2, d2u_dy2
    end

    # Convection-diffusion equation
    function f_cd(u,
            t,
            ddx_dy = first_derivative,
            d2dx2_d2dy2 = second_derivative,
            viscosity = viscosity,
            speed = speed,
            dx = dx,
            dy = dy)
        du_dx, du_dy = ddx_ddy(u, dx, dy)
        d2u_dx2, d2u_dy2 = d2dx2_d2dy2(u, dx, dy)
        return -speed[1] * du_dx - speed[2] * du_dy .+ viscosity[1] * d2u_dx2 .+
               viscosity[2] * d2u_dy2
    end
    return f_cd
end

# So this is the force
f_cd(u) = gen_conv_diff_f([0.1, 0.1], [0.00001, 0.00001], dx, dy)

# create the NN
NN = create_nn_cd()

# * We create the right hand side of the NODE, by combining the NN with f_u 
f_NODE = create_NODE_cd(NN, p; is_closed = true)
# and get the parametrs that you want to train
θ, st = Lux.setup(rng, f_NODE)
