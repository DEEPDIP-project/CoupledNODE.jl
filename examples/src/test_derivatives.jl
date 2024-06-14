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
        return d2u[2:(end - 1), 2:(end - 1), :] / Δx2# + Δy2)
    elseif dims == 3
        d2u[:, :, 2:(end - 1), :] += (up[:, :, 3:end, :] - 2 * up[:, :, 2:(end - 1), :] +
                                      up[:, :, 1:(end - 2), :])
        return d2u[2:(end - 1), 2:(end - 1), 2:(end - 1), :] / Δx2# + Δy2 + Δz2)
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

        @. ((upx - 2 * u + umx) + (upy - 2 * u + umy)) / Δx2# + Δy2)
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

        @. ((upx - 2 * u + umx) + (upy - 2 * u + umy)) / Δx2# + Δy2)
    end
end

using Pkg
# This fork here claims to have a CUDA compatible version of ShiftedArrays, but it does not work without using collect() thus hindering the performance
#Pkg.add(url="https://github.com/RainerHeintzmann/ShiftedArrays.jl.git")
# this is to use the original ShiftedArrays
#Pkg.add(url="https://github.com/JuliaArrays/ShiftedArrays.jl")
using ShiftedArrays
function Laplacian4(u, Δx2, Δy2 = 0.0, Δz2 = 0.0)
    dims = ndims(u) - 1  # Subtract 1 for the batch dimension

    if dims == 2
        upx = collect(ShiftedArrays.circshift(u, (1, 0, 0)))
        upy = collect(ShiftedArrays.circshift(u, (0, 1, 0)))
        umx = collect(ShiftedArrays.circshift(u, (-1, 0, 0)))
        umy = collect(ShiftedArrays.circshift(u, (0, -1, 0)))

        #@. (upx + umx + upy + umy - u * 4) / (Δx2 + Δy2)
        @. (upx + umx + upy + umy - u * 4) / Δx2# + Δy2)
    end
end

a = ShiftedArrays.circshift(u_initial, (1, 0, 0))
b = ShiftedArrays.circshift(u_initial, (0, 1, 0))
collect(a) + collect(b) + collect(u_initial)

###############

using Lux, Flux
rng = Random.seed!(1234);
O = zeros(size(u_initial)[1]+2, size(u_initial)[2]+2, 1, size(u_initial)[3])
M = Lux.Chain( 
#    # Pad for PBC
#    WrappedFunction(x -> NNlib.pad_circular(x, 1; dims=[1,2])), 
#    # reshape to add the channel dimension (and do not forget the padding)
#    ReshapeLayer((nux + 2, nuy+ 2, 1)),
#    x -> let x=x
#        O[2:end-1, 2:end-1, 1, :] .= x
#        O[1, 2:end-1, 1, :] .= x[end, :,:]
#        O[end, 2:end-1, 1, :] .= x[1, :, :]
#        O[2:end-1, 1, 1, :] .= x[:, end, :]
#        O[2:end-1, end, 1,:] .= x[:, 1, :]
#        O
#    end,
    WrappedFunction(x -> NNlib.pad_circular(x, 1; dims=[1,2])), 
    x -> Flux.unsqueeze(x, 3),
#    x -> NNlib.pad_circular(reshape(x, size(x, 1), size(x, 2), 1, size(x, 3)), 1; dims=(1, 2)),
    Lux.Conv((3,3),1=>1, use_bias=false, allow_fast_activation=false, pad=(0,0), stride=(1,1)),
    SelectDim(3, 1),
)
θ, st = Lux.setup(rng, M);
using ComponentArrays
θ = ComponentArrays.ComponentArray(θ)
θ.layer_3.weight = [
    0.0f0 1.0f0 0.0f0;
    1.0f0 -4.0f0 1.0f0;
    0.0f0 1.0f0 0.0f0
]
function Laplacian_Lux(u, Δx2, Δy2 = 0.0, Δz2 = 0.0)
    Lux.apply(M, u, θ, st)[1]
end


###############
N = nux
using LinearAlgebra
K = Array(Tridiagonal([1.0f0 for i in 1:(N - 1)], [-4.0f0 for i in 1:N],
    [1.0f0 for i in 1:(N - 1)]))
# Use periodic boundary conditions
K[1, end] = 1.0f0
K[end, 1] = 1.0f0
K

# CONVOLUTION != MATRIX MULTIPLICATION
# here you have to do like this [link](https://docs.sciml.ai/DiffEqDocs/stable/tutorials/faster_ode_example/#Example-Accelerating-Linear-Algebra-PDE-Semi-Discretization)

function Laplacian_stencil(u, Δx2, Δy2 = 0.0, Δz2 = 0.0)
    #permutedims(K .* u, (2, 1, 3))
    K .* u
end




###############
using SparseArrays
Δz² = dux^2
n=nux
∇²_op = [1 / Δz², -2 / Δz², 1 / Δz²]; 
FT = MY_TYPE
D = Tridiagonal(ones(FT, n+1) .* 1,
    ones(FT, n + 2) .* -4,
    ones(FT, n+1) .* 1)




yy = zeros(size(u_initial)[1]+2, size(u_initial)[2]+2, size(u_initial)[3])
function Laplacian_stencil2(u, Δx2, Δy2 = 0.0, Δz2 = 0.0)
    # Create the ghost grid
    yy[2:end-1, 2:end-1, :] .= u
    yy[1, 2:end-1,:] .= x[end, :,:]
    yy[end, 2:end-1,:] .= x[1, :, :]
    yy[2:end-1, 1, :] .= x[:, end, :]
    yy[2:end-1, end, :] .= x[:, 1, :]
    permutedims((D .* yy)[2:end-1, 2:end-1, :], (2,1,3))
end


############



@time F_old(u, v) = D_u * Old_Laplacian(u, grid_GS_u.dx, grid_GS_u.dy) .- u .* v .^ 2 .+
                    f .* (1.0f0 .- u)
@time F_new2(u, v) = D_u * Laplacian2(u, grid_GS_u.dx, grid_GS_u.dy) .- u .* v .^ 2 .+
                     f .* (1.0f0 .- u)
@time F_new3(u, v) = D_u * Laplacian3(u, grid_GS_u.dx, grid_GS_u.dy) .- u .* v .^ 2 .+
                     f .* (1.0f0 .- u)
@time F_new4(u, v) = D_u * Laplacian4(u, grid_GS_u.dx, grid_GS_u.dy) .- u .* v .^ 2 .+
                     f .* (1.0f0 .- u)
@time F_Lux(u, v) = D_u * Laplacian_Lux(u, grid_GS_u.dx, grid_GS_u.dy) .- u .* v .^ 2 .+
                     f .* (1.0f0 .- u)
@time F_s(u, v) = D_u * Laplacian_stencil(u, grid_GS_u.dx, grid_GS_u.dy) .- u .* v .^ 2 .+
                     f .* (1.0f0 .- u)
@time F_s2(u, v) = D_u * Laplacian_stencil2(u, grid_GS_u.dx, grid_GS_u.dy) .- u .* v .^ 2 .+
                     f .* (1.0f0 .- u)



                
GC.gc()
@time o = F_old(grid_GS_v.grid_data, grid_GS_v.grid_data);
GC.gc()
@time n2 = F_new2(grid_GS_v.grid_data, grid_GS_v.grid_data);
GC.gc()
@time n3 = F_new3(grid_GS_v.grid_data, grid_GS_v.grid_data);
GC.gc()
@time n4 = F_new4(grid_GS_v.grid_data, grid_GS_v.grid_data);
GC.gc()
@time nl = F_Lux(grid_GS_v.grid_data, grid_GS_v.grid_data);
GC.gc()
@time ns = F_s(grid_GS_v.grid_data, grid_GS_v.grid_data);
GC.gc()
@time ns2 = F_s2(grid_GS_v.grid_data, grid_GS_v.grid_data);

o ≈ n2
o ≈ n3
o ≈ n4
o ≈ nl
o ≈ ns
o ≈ ns2

o

sum(o)
sum(ns)
sum(ns2)
o
ns

