"""
    struct Grid

Object containing the grid information. It can handle 1D, 2D, and 3D grids.
"""
#TODO: does not need to store data
#TODO: can we make this the same struct as 'params' in NavierStokes.jl??
Base.@kwdef mutable struct Grid
    const dim::Int
    const dtype::DataType = Float32
    const dx::Union{Float32, Float64}
    const dy::Union{Float32, Float64} = 0.0f0
    const dz::Union{Float32, Float64} = 0.0f0
    const nx::Int
    const ny::Int = 0
    const nz::Int = 0
    const N::Int
    const x::Union{Vector{Float32}, Vector{Float64}, Nothing}
    const y::Union{Vector{Float32}, Vector{Float64}, Nothing}
    const z::Union{Vector{Float32}, Vector{Float64}, Nothing}
    nsim::Int = 1
    grid_data::Any
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
           g1.N == g2.N &&
           g1.x == g2.x &&
           g1.y == g2.y &&
           g1.z == g2.z &&
           g1.nsim == g2.nsim &&
           g1.grid_data == g2.grid_data
end

function make_grid(; dim::Int, dtype::DataType, dx::Union{Float32, Float64}, nx::Int,
        dy::Union{Float32, Float64} = 0.0f0, ny::Int = 0,
        dz::Union{Float32, Float64} = 0.0f0, nz::Int = 0,
        nsim::Int = 1, grid_data = nothing)

    # Check that the data type is the same as the specified data type
    if (grid_data !== nothing && eltype(grid_data) != dtype)
        throw(ArgumentError("The data type is not the same as the specified data type."))
    end

    # Initialize the grid dimensions
    N = nx * max(1, ny) * max(1, nz)
    x = collect(0:dx:((nx - 1) * dx))
    y = dy == 0 ? nothing : collect(0:dy:((ny - 1) * dy))
    z = dz == 0 ? nothing : collect(0:dz:((nz - 1) * dz))

    # Extract dimensions from grid_data
    dims = size(grid_data)
    # Extract dimensions from grid_data
    dims = size(grid_data)
    ndims = length(dims)

    nsim = dims[end]  # nsim is always the last dimension
    nx = dims[1]  # nx is always the first dimension
    # ny and nz are optional
    ny = ndims > 2 ? dims[2] : 0
    nz = ndims > 3 ? dims[3] : 0

    # Validate dimensions
    if nx != size(grid_data)[1] || (ny != 0 && ny != size(grid_data)[2]) ||
       (nz != 0 && nz != size(grid_data)[3])
        throw(DimensionMismatch("The grid data dimensions (nx, ny, nz) = ($(nx), $(ny), $(nz)) do not match the expected dimensions ($(size(grid_data)[1:end-1]))"))
    end

    return Grid(dim, dtype, dx, dy, dz, nx, ny, nz, N, x, y, z, nsim, grid_data)
end
