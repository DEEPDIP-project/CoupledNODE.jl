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
    dtype::DataType = Float32
    dx::Union{Float32, Float64}
    dy::Union{Float32, Float64} = 0.0f0
    dz::Union{Float32, Float64} = 0.0f0
    nx::Int
    ny::Int = 0
    nz::Int = 0
    nsim::Int = 1
    N::Int
    x::Union{Vector{Float32}, Vector{Float64}, Nothing}
    y::Union{Vector{Float32}, Vector{Float64}, Nothing}
    z::Union{Vector{Float32}, Vector{Float64}, Nothing}
    grid_data::Any
    linear_data::Any
end

import Base: ==
function ==(g1::Grid, g2::Grid)
    return g1.dim == g2.dim &&
           g1.dtype == g2.dtype &&
           g1.dx == g2.dx &&
           g1.dy == g2.dy &&
           g1.dz == g2.dz &&
           g1.nx == g2.nx &&
           g1.ny == g2.ny &&
           g1.nz == g2.nz &&
           g1.nsim == g2.nsim &&
           g1.N == g2.N &&
           g1.x == g2.x &&
           g1.y == g2.y &&
           g1.z == g2.z &&
           g1.grid_data == g2.grid_data &&
           g1.linear_data == g2.linear_data
end

function make_grid(; dim::Int, dtype::DataType, dx::Union{Float32, Float64}, nx::Int,
                   dy::Union{Float32, Float64} = 0.0f0, ny::Int = 0,
                   dz::Union{Float32, Float64} = 0.0f0, nz::Int = 0,
                   nsim::Int = 1, grid_data=nothing, linear_data=nothing)
    
    # Check that only one data source is provided
    if grid_data === nothing && linear_data === nothing
        throw(ArgumentError("At least one of grid_data or linear_data must be provided."))
    elseif grid_data !== nothing && linear_data !== nothing
        throw(ArgumentError("Only one of grid_data or linear_data can be provided."))
    end
    # and check that the data type is the same as the specified data type
    if (grid_data !== nothing && eltype(grid_data) != dtype) || (linear_data !== nothing && eltype(linear_data) != dtype)
        throw(ArgumentError("The data type is not the same as the specified data type."))
    end

    # Initialize the grid dimensions
    N = nx * max(1, ny) * max(1, nz)
    x = collect(0:dx:((nx - 1) * dx))
    y = dy == 0 ? nothing : collect(0:dy:((ny - 1) * dy))
    z = dz == 0 ? nothing : collect(0:dz:((nz - 1) * dz))

    # Check from which data you have to initialize the grid
    if linear_data === nothing
        if nz > 0
            linear_data = view(grid_data, :, :, :, :)
            linear_data = reshape(linear_data, nx*ny*nz, nsim)
        elseif ny > 0
            linear_data = view(grid_data, :, :, :)
            linear_data = reshape(linear_data, nx*ny, nsim)
        else
            linear_data = view(grid_data, :, :)
            linear_data = reshape(linear_data, nx, nsim)
        end
    else
        grid_data = view(linear_data, :)
        if nz > 0
            grid_data = reshape(grid_data, nx, ny, nz, nsim)
        elseif ny > 0
            grid_data = reshape(grid_data, nx, ny, nsim)
        else
            grid_data = reshape(grid_data, nx, nsim)
        end
    end

    return Grid(dim, dtype, dx, dy, dz, nx, ny, nz, nsim, N, x, y, z, grid_data, linear_data)
end
