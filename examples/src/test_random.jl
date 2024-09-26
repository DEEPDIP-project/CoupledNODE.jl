using OffsetArrays
using NNlib
using BenchmarkTools
using Plots

# Define the Laplacian using views
function laplacian_views(u, h_x, h_y)
    nx, ny = size(u)
    Δu = zeros(nx, ny)

    @views for i in 2:(nx - 1)
        for j in 2:(ny - 1)
            Δu[i, j] = (u[i - 1, j] - 2u[i, j] + u[i + 1, j]) / h_x^2 +
                       (u[i, j - 1] - 2u[i, j] + u[i, j + 1]) / h_y^2
        end
    end

    return Δu
end

# Define the Laplacian using convolutions
function laplacian_convolution(u, h_x, h_y)
    kernel = [0 1 0;
              1 -4 1;
              0 1 0]
    Δu = conv(u, kernel, pad = 1)
    Δu .= Δu / (h_x^2 * h_y^2)  # Adjust for grid spacings
    return Δu
end

# Define the grid size and spacings
nx, ny = 100, 100
h_x = 1.0
h_y = 2.0

# Initialize the grid function u
u = zeros(nx, ny)
u[50, 50] = 1.0  # A delta function at the center

# Benchmark the views method
@btime laplacian_views($u, $h_x, $h_y)

# Benchmark the convolution method
@btime laplacian_convolution($u, $h_x, $h_y)

# Apply the Laplacian using the convolution method and visualize the result
Δu = laplacian_convolution(u, h_x, h_y)
heatmap(
    Δu, title = "Discrete Laplace Operator Using Convolution", xlabel = "x", ylabel = "y")
