"""
    struct Grid

This object contains the grid information.

Fields:
- `dux::Float64`: The grid spacing in the x-direction for u.
- `duy::Float64`: The grid spacing in the y-direction for u.
- `nux::Int`: The number of grid points in the x-direction for u.
- `nuy::Int`: The number of grid points in the y-direction for u.
- `dvx::Float64`: The grid spacing in the x-direction for v.
- `dvy::Float64`: The grid spacing in the y-direction for v.
- `nvx::Int`: The number of grid points in the x-direction for v.
- `nvy::Int`: The number of grid points in the y-direction for v.
- `Nu::Int`: The total number of elements for u.
- `Nv::Int`: The total number of elements for v.

Constructor:
- `Grid(dux::Float64, duy::Float64, nux::Int, nuy::Int, dvx::Float64, dvy::Float64, nvx::Int, nvy::Int)`: Constructs a `Grid` object with the given grid parameters.
"""
struct Grid
    dux::Union{Float32, Float64}
    duy::Union{Float32, Float64}
    nux::Int
    nuy::Int
    dvx::Union{Float32, Float64}
    dvy::Union{Float32, Float64}
    nvx::Int
    nvy::Int
    Nu::Int
    Nv::Int

    function Grid(dux::Union{Float32, Float64},
            duy::Union{Float32, Float64},
            nux::Int,
            nuy::Int,
            dvx::Union{Float32, Float64},
            dvy::Union{Float32, Float64},
            nvx::Int,
            nvy::Int;
            convert_to_float32::Bool = false)
        Nu = nux * nuy
        Nv = nvx * nvy
        if convert_to_float32
            new(Float32(dux), Float32(duy), nux, nuy,
                Float32(dvx), Float32(dvy), nvx, nvy, Nu, Nv)
        else
            new(dux, duy, nux, nuy, dvx, dvy, nvx, nvy, Nu, Nv)
        end
    end
end
