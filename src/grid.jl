"""
    struct Grid

Object containing the grid information. It can handle 1D, 2D, and 3D grids.

Fields:
- `dim::Int`: The dimensionality of the grid.
- `dx::Union{Float32, Float64}`: The grid spacing in the x-direction.
- `dy::Union{Float32, Float64}`: The grid spacing in the y-direction. Default is 0 for 1D grid.
- `dz::Union{Float32, Float64}`: The grid spacing in the z-direction. Default is 0 for 1D and 2D grids.
- `nx::Int`: The number of grid points in the x-direction.
- `ny::Int`: The number of grid points in the y-direction. Default is 0 for 1D grid.
- `nz::Int`: The number of grid points in the z-direction. Default is 0 for 1D and 2D grids.
- `N::Int`: The total number of elements in the grid.

Constructor:
- `Grid(dim::Int, dx::Union{Float32, Float64}, dy::Union{Float32, Float64} = 0, dz::Union{Float32, Float64} = 0, nx::Int, ny::Int = 0, nz::Int = 0, convert_to_float32::Bool = false)`: Constructs a `Grid` object with the given grid parameters. The `dy`, `dz`, `ny`, and `nz` parameters can be omitted for 1D and 2D grids. 
"""
Base.@kwdef struct Grid
    dim::Int
    dx::Union{Float32, Float64}
    dy::Union{Float32, Float64} = 0.0f0
    dz::Union{Float32, Float64} = 0.0f0
    nx::Int
    ny::Int = 0
    nz::Int = 0
    nsim::Int = 1
    grid_data::Any
    N::Int
    x::Union{Vector{Float32}, Vector{Float64}, Nothing}
    y::Union{Vector{Float32}, Vector{Float64}, Nothing}
    z::Union{Vector{Float32}, Vector{Float64}, Nothing}
    linear_data::Any
end

function make_grid(; dim::Int, dx::Union{Float32, Float64}, nx::Int,
                   dy::Union{Float32, Float64} = 0.0f0, ny::Int = 0,
                   dz::Union{Float32, Float64} = 0.0f0, nz::Int = 0,
                   nsim::Int = 1, grid_data)
    
    N = nx * max(1, ny) * max(1, nz)
    x = collect(0:dx:((nx - 1) * dx))
    y = dy == 0 ? nothing : collect(0:dy:((ny - 1) * dy))
    z = dz == 0 ? nothing : collect(0:dz:((nz - 1) * dz))

    if nz > 0
        linear_data = zeros(eltype(grid_data),nx*ny*nz, nsim)
    elseif ny > 0
        linear_data = zeros(eltype(grid_data),nx*ny, nsim)
    else
        linear_data = zeros(eltype(grid_data), nx, nsim)
    end

    return Grid(dim, dx, dy, dz, nx, ny, nz, nsim, grid_data, N, x, y, z, linear_data)
end

function linear_to_grid(grid::Grid)
    grid.grid_data[:] .= grid.linear_data[:]
end
function grid_to_linear(grid::Grid)
    grid.linear_data[:] .= grid.grid_data[:]
end