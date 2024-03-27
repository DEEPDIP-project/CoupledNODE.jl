using Lux
using NaNMath
using SciMLSensitivity
using DiffEqFlux
using DifferentialEquations
using Plots
using Zygote
using Random
rng = Random.seed!(1234)
using OptimizationOptimisers
using Statistics
using ComponentArrays
using CUDA
ArrayType = CUDA.functional() ? CuArray : Array

# Import our custom backend functions
include("coupling_functions/functions_example.jl")
include("coupling_functions/functions_NODE.jl")
include("coupling_functions/functions_loss.jl")

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
