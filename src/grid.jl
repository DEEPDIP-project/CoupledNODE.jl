"""
    struct Grid

This object contains the grid information. It can handle 1D, 2D, and 3D grids.

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
- `Grid(dim::Int, dx::Union{Float32, Float64}, dy::Union{Float32, Float64} = 0, dz::Union{Float32, Float64} = 0, nx::Int, ny::Int = 0, nz::Int = 0, convert_to_float32::Bool = false)`: Constructs a `Grid` object with the given grid parameters. The `dy`, `dz`, `ny`, and `nz` parameters can be omitted for 1D and 2D grids. If `convert_to_float32` is `true`, the grid spacings are converted to `Float32`.
"""
Base.@kwdef struct Grid
    dim::Int
    dx::Union{Float32, Float64}
    dy::Union{Float32, Float64} = 0.0f0
    dz::Union{Float32, Float64} = 0.0f0
    nx::Int
    ny::Int = 0
    nz::Int = 0
    N::Int = nx * max(1, ny) * max(1, nz)
    x::Union{Vector{Float32}, Vector{Float64}} = collect(0:dx:((nx - 1) * dx))
    y::Union{Vector{Float32}, Vector{Float64}, Nothing} = dy == 0 ? nothing :
                                                          collect(0:dy:((ny - 1) * dy))
    z::Union{Vector{Float32}, Vector{Float64}, Nothing} = dz == 0 ? nothing :
                                                          collect(0:dz:((nz - 1) * dz))
end

function linear_to_grid(g::Grid, u)
    if g.dim == 1
        return reshape(u, g.nx, size(u)[end])
    elseif g.dim == 2
        return reshape(u, g.nx, g.ny, size(u)[end])
    elseif g.dim == 3
        return reshape(u, g.nx, g.ny, g.nz, size(u)[end])
    end
end

function grid_to_linear(g::Grid, u)
    if g.dim == 1
        return reshape(u, g.N, size(u)[end])
    elseif g.dim == 2
        return reshape(u, g.N, size(u)[end])
    elseif g.dim == 3
        return reshape(u, g.N, size(u)[end])
    end
end
