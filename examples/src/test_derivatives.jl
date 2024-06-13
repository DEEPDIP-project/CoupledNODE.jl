const MY_TYPE = Float32 # use float32 if you plan to use a GPU

include("./../../src/grid.jl")
dux = duy = dvx = dvy = 1.0f0
nux = nuy = nvx = nvy = 1024

# Definition of the initial condition as a random perturbation over a constant background to add variety. 
import Random
function initial_condition(U₀, V₀, ε_u, ε_v; nsimulations = 1)
    u_init = U₀ .+ ε_u .* Random.randn(Float32, nux, nuy, nsimulations)
    v_init = V₀ .+ ε_v .* Random.randn(Float32, nvx, nvy, nsimulations)
    return u_init, v_init
end
U₀ = 0.5f0    # initial concentration of u
V₀ = 0.25f0   # initial concentration of v
ε_u = 0.05f0  # magnitude of the perturbation on u
ε_v = 0.1f0   # magnitude of the perturbation on v
nsim = 10
u_initial, v_initial = initial_condition(U₀, V₀, ε_u, ε_v, nsimulations = nsim);

# Declare the grid object 
grid_GS_u = make_grid(dim = 2, dtype = MY_TYPE, dx = dux, nx = nux, dy = duy,
    ny = nuy, nsim = nsim, grid_data = u_initial)
grid_GS_v = make_grid(dim = 2, dtype = MY_TYPE, dx = dvx, nx = nvx, dy = dvy,
    ny = nvy, nsim = nsim, grid_data = v_initial)

# These are the GS parameters (also used in example 02.01) that we will try to learn
D_u = 0.16f0
D_v = 0.08f0
f = 0.055f0
k = 0.062f0;

# OLD Laplacian
function add_dim_1(x)
    return reshape(x, 1, size(x)...)
end
function add_dim_2(x)
    s = size(x)
    return reshape(x, s[1], 1, s[2:end]...)
end
function add_dim_3(x)
    s = size(x)
    return reshape(x, s[1:2]..., 1, s[3:end]...)
end
function circular_pad(u, dims)
    if dims == 1
        u_padded = vcat(add_dim_1(u[end, :]), u, add_dim_1(u[1, :]))
    elseif dims == 2
        u_padded = vcat(add_dim_1(u[end, :, :]), u, add_dim_1(u[1, :, :]))
        u_padded = hcat(
            add_dim_2(u_padded[:, end, :]), u_padded, add_dim_2(u_padded[:, 1, :]))
    elseif dims == 3
        u_padded = vcat(add_dim_1(u[end, :, :, :]), u, add_dim_1(u[1, :, :, :]))
        u_padded = hcat(
            add_dim_2(u_padded[:, end, :, :]), u_padded, add_dim_2(u_padded[:, 1, :, :]))
        u_padded = cat(add_dim_3(u_padded[:, :, end, :]), u_padded,
            add_dim_3(u_padded[:, :, 1, :]), dims = 3)
    end
    return u_padded
end
function Old_Laplacian(u, Δx2, Δy2 = 0.0, Δz2 = 0.0)
    dims = ndims(u) - 1  # Subtract 1 for the batch dimension
    up = circular_pad(u, dims)
    d2u = similar(up)

    if dims == 1
        d2u[2:(end - 1), :] = (up[3:end, :] - 2 * up[2:(end - 1), :] +
                               up[1:(end - 2), :])

        return d2u[2:(end - 1), :] / Δx2
    elseif dims == 2
        d2u[2:(end - 1), :, :] = (up[3:end, :, :] - 2 * up[2:(end - 1), :, :] +
                                  up[1:(end - 2), :, :])
        d2u[:, 2:(end - 1), :] += (up[:, 3:end, :] - 2 * up[:, 2:(end - 1), :] +
                                   up[:, 1:(end - 2), :])
        return d2u[2:(end - 1), 2:(end - 1), :] / (Δx2 + Δy2)
    elseif dims == 3
        d2u[:, :, 2:(end - 1), :] += (up[:, :, 3:end, :] - 2 * up[:, :, 2:(end - 1), :] +
                                      up[:, :, 1:(end - 2), :])
        return d2u[2:(end - 1), 2:(end - 1), 2:(end - 1), :] / (Δx2 + Δy2 + Δz2)
    end
end
###########

function Laplacian2(u, Δx2, Δy2 = 0.0, Δz2 = 0.0)
    dims = ndims(u) - 1  # Subtract 1 for the batch dimension

    if dims == 2
        upx = circshift(u, (1, 0, 0))
        umx = circshift(u, (0, 1, 0))
        upy = circshift(u, (-1, 0, 0))
        umy = circshift(u, (0, -1, 0))

        @. ((upx - 2 * u + umx) + (upy - 2 * u + umy)) / (Δx2 + Δy2)
    end
end

function Laplacian3(u, Δx2, Δy2 = 0.0, Δz2 = 0.0)
    dims = ndims(u) - 1  # Subtract 1 for the batch dimension

    if dims == 2
        upx = similar(u)
        umx = similar(u)
        upy = similar(u)
        umy = similar(u)
        circshift!(upx, u, (1, 0, 0))
        circshift!(upy, u, (0, 1, 0))
        circshift!(umx, u, (-1, 0, 0))
        circshift!(umy, u, (0, -1, 0))

        @. ((upx - 2 * u + umx) + (upy - 2 * u + umy)) / (Δx2 + Δy2)
    end
end

using ShiftedArrays
function Laplacian4(u, Δx2, Δy2 = 0.0, Δz2 = 0.0)
    dims = ndims(u) - 1  # Subtract 1 for the batch dimension

    if dims == 2
        upx = ShiftedArrays.circshift(u, (1, 0, 0))
        upy = ShiftedArrays.circshift(u, (0, 1, 0))
        umx = ShiftedArrays.circshift(u, (-1, 0, 0))
        umy = ShiftedArrays.circshift(u, (0, -1, 0))

        @. (upx + umx + upy + umy - u * 4) / (Δx2 + Δy2)
    end
end

@time F_old(u, v) = D_u * Old_Laplacian(u, grid_GS_u.dx, grid_GS_u.dy) .- u .* v .^ 2 .+
                    f .* (1.0f0 .- u)
@time F_new2(u, v) = D_u * Laplacian2(u, grid_GS_u.dx, grid_GS_u.dy) .- u .* v .^ 2 .+
                     f .* (1.0f0 .- u)
@time F_new3(u, v) = D_u * Laplacian3(u, grid_GS_u.dx, grid_GS_u.dy) .- u .* v .^ 2 .+
                     f .* (1.0f0 .- u)
@time F_new4(u, v) = D_u * Laplacian4(u, grid_GS_u.dx, grid_GS_u.dy) .- u .* v .^ 2 .+
                     f .* (1.0f0 .- u)

@time o = F_old(grid_GS_v.grid_data, grid_GS_v.grid_data);
@time n2 = F_new2(grid_GS_v.grid_data, grid_GS_v.grid_data);
@time n3 = F_new3(grid_GS_v.grid_data, grid_GS_v.grid_data);
@time n4 = F_new4(grid_GS_v.grid_data, grid_GS_v.grid_data);

o ≈ n2
o ≈ n3
o ≈ n4
