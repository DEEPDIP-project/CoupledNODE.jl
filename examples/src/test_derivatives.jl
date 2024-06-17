const MY_TYPE = Float32 # use float32 if you plan to use a GPU

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

# These are the GS parameters (also used in example 02.01) that we will try to learn
D_u = 0.16f0
D_v = 0.08f0
f = 0.055f0
k = 0.062f0;

# Declare the grid object 
include("./../../src/grid.jl")
dux = duy = dvx = dvy = 1.0f0
nux = nuy = nvx = nvy = 1024
nsim = 10
u_initial, v_initial = initial_condition(U₀, V₀, ε_u, ε_v, nsimulations = nsim);
grid_GS_u = make_grid(dim = 2, dtype = MY_TYPE, dx = dux, nx = nux, dy = duy,
    ny = nuy, nsim = nsim, grid_data = u_initial)
grid_GS_v = make_grid(dim = 2, dtype = MY_TYPE, dx = dvx, nx = nvx, dy = dvy,
    ny = nvy, nsim = nsim, grid_data = v_initial)

#####################   DIFFERENT LAPLACIAN IMPLEMENTATIONS   #####################
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
    u_padded = vcat(add_dim_1(u[end, :, :]), u, add_dim_1(u[1, :, :]))
    u_padded = hcat(
        add_dim_2(u_padded[:, end, :]), u_padded, add_dim_2(u_padded[:, 1, :]))
end
function Old_Laplacian(u, Δx2, Δy2 = 0.0, Δz2 = 0.0)
    dims = ndims(u) - 1  # Subtract 1 for the batch dimension
    up = circular_pad(u, dims)
    d2u = similar(up)

    d2u[2:(end - 1), :, :] = (up[3:end, :, :] - 2 * up[2:(end - 1), :, :] +
                              up[1:(end - 2), :, :])
    d2u[:, 2:(end - 1), :] += (up[:, 3:end, :] - 2 * up[:, 2:(end - 1), :] +
                               up[:, 1:(end - 2), :])
    return d2u[2:(end - 1), 2:(end - 1), :] / Δx2# + Δy2)
end

###########
# this uses standard circshift
function Laplacian_c1(u, Δx2, Δy2 = 0.0, Δz2 = 0.0)
    upx = circshift(u, (1, 0, 0))
    umx = circshift(u, (0, 1, 0))
    upy = circshift(u, (-1, 0, 0))
    umy = circshift(u, (0, -1, 0))

    @. ((upx - 2 * u + umx) + (upy - 2 * u + umy)) / Δx2# + Δy2)
end

###########
# this uses circshift! to avoid memory allocation
function Laplacian_c2(u, Δx2, Δy2 = 0.0, Δz2 = 0.0)
    circshift!(upx0, u, (1, 0, 0))
    circshift!(upy0, u, (0, 1, 0))
    circshift!(umx0, u, (-1, 0, 0))
    circshift!(umy0, u, (0, -1, 0))

    @. ((upx0 - 2 * u + umx0) + (upy0 - 2 * u + umy0)) / Δx2
end

###########
# this uses ShiftedArrays
# [!] however, this is not compatible with CUDA
using ShiftedArrays
function Laplacian_c3(u, Δx2, Δy2 = 0.0, Δz2 = 0.0)
    upx = ShiftedArrays.circshift(u, (1, 0, 0))
    upy = ShiftedArrays.circshift(u, (0, 1, 0))
    umx = ShiftedArrays.circshift(u, (-1, 0, 0))
    umy = ShiftedArrays.circshift(u, (0, -1, 0))

    @. (upx + umx + upy + umy - u * 4) / Δx2
end

###########
# this uses a fork from "https://github.com/RainerHeintzmann/ShiftedArrays.jl.git"
# that claims to make ShiftedArrays compatible with CUDA
# however you have to call collect() to make it work, thus hindering the performance
# (on GPU it has to be tested)
#using Pkg
#Pkg.add(url="https://github.com/DEEPDIP-project/cShiftedArrays.jl.git")
#Pkg.add(url="https://github.com/JuliaArrays/ShiftedArrays.jl.git")
using cShiftedArrays
function Laplacian_c4(u, Δx2, Δy2 = 0.0, Δz2 = 0.0)
    upx = collect(cShiftedArrays.circshift(u, (1, 0, 0)))
    upy = collect(cShiftedArrays.circshift(u, (0, 1, 0)))
    umx = collect(cShiftedArrays.circshift(u, (-1, 0, 0)))
    umy = collect(cShiftedArrays.circshift(u, (0, -1, 0)))

    @. (upx + umx + upy + umy - u * 4) / Δx2# + Δy2)
end

###############
# This usese the Lux library and a convolutional layer
using Lux, Flux
using MLUtils
rng = Random.seed!(1234);
M = Lux.Chain(
    # Pad for PBC
    x -> begin
        @views uu2[2:(end - 1), 2:(end - 1), 1, :] .= x
        @views uu2[1, 2:(end - 1), 1, :] .= x[end, :, :]
        @views uu2[end, 2:(end - 1), 1, :] .= x[1, :, :]
        @views uu2[2:(end - 1), 1, 1, :] .= x[:, end, :]
        @views uu2[2:(end - 1), end, 1, :] .= x[:, 1, :]
        uu2
    end,
    Lux.Conv((3, 3), 1 => 1, use_bias = false),
    SelectDim(3, 1)
)
θ, st = Lux.setup(rng, M);
using ComponentArrays
θ = ComponentArrays.ComponentArray(θ)
θ.layer_2.weight = [0.0f0 1.0f0 0.0f0;
                    1.0f0 -4.0f0 1.0f0;
                    0.0f0 1.0f0 0.0f0]
function Laplacian_Lux(u, Δx2, Δy2 = 0.0, Δz2 = 0.0)
    Lux.apply(M, u, θ, st)[1]
end

###############
# this is a Lux variant with padding instead of preallocation
Mpd = Lux.Chain(
    # Pad for PBC
    WrappedFunction(x -> unsqueeze(NNlib.pad_circular(x, 1; dims = [1, 2]), 3)),
    Lux.Conv((3, 3), 1 => 1, use_bias = false),
    SelectDim(3, 1)
)
θpd, stpd = Lux.setup(rng, Mpd);
θpd = ComponentArrays.ComponentArray(θ)
θpd.layer_2.weight = [0.0f0 1.0f0 0.0f0;
                      1.0f0 -4.0f0 1.0f0;
                      0.0f0 1.0f0 0.0f0]
function Laplacian_Lux_pd(u, Δx2, Δy2 = 0.0, Δz2 = 0.0)
    Lux.apply(Mpd, u, θpd, stpd)[1]
end

###############
# This one uses a stencil and matrix multiplication
# similarly to [this](https://docs.sciml.ai/DiffEqDocs/stable/tutorials/faster_ode_example/#Example-Accelerating-Linear-Algebra-PDE-Semi-Discretization)
N = nux
using LinearAlgebra#, Kronecker
K = Array(Tridiagonal([1.0f0 for i in 1:(N - 1)], [-2.0f0 for i in 1:N],
    [1.0f0 for i in 1:(N - 1)]))
# Use periodic boundary conditions
K[end, 1] = 1.0
K[1, end] = 1.0
function Laplacian_stencil(u, Δx2, Δy2 = 0.0, Δz2 = 0.0)
    NNlib.batched_mul(K, u_initial) + NNlib.batched_mul(u_initial, K)
end

###############
# This one uses convolution
FT = MY_TYPE
n = nux
D = [0.0f0 1.0f0 0.0f0;
     1.0f0 -4.0f0 1.0f0;
     0.0f0 1.0f0 0.0f0]
D_3D = reshape(D, (3, 3, 1, 1))
function Laplacian_conv(u, Δx2, Δy2 = 0.0, Δz2 = 0.0)
    # Create the ghost grid
    uu[2:(end - 1), 2:(end - 1), :] .= u
    uu[1, 2:(end - 1), :] .= u[end, :, :]
    uu[end, 2:(end - 1), :] .= u[1, :, :]
    uu[2:(end - 1), 1, :] .= u[:, end, :]
    uu[2:(end - 1), end, :] .= u[:, 1, :]
    dropdims(NNlib.conv(unsqueeze(uu, 3), D_3D), dims = 3)
end

#############

# Declare the grid object 
grid_GS_u = make_grid(dim = 2, dtype = MY_TYPE, dx = dux, nx = nux, dy = duy,
    ny = nuy, nsim = nsim, grid_data = u_initial)
grid_GS_v = make_grid(dim = 2, dtype = MY_TYPE, dx = dvx, nx = nvx, dy = dvy,
    ny = nvy, nsim = nsim, grid_data = v_initial)

# Allocate some memory for some of the algos
upx0 = similar(u_initial);
umx0 = similar(u_initial);
upy0 = similar(u_initial);
umy0 = similar(u_initial);
uu = zeros(MY_TYPE, size(u_initial)[1] + 2, size(u_initial)[2] + 2, size(u_initial)[3]);
uu2 = zeros(MY_TYPE, size(u_initial)[1] + 2, size(u_initial)[2] + 2, 1, size(u_initial)[3]);

# Precompile all the functions
function_list = [
    Old_Laplacian,
    Laplacian_c1,
    Laplacian_c2,
    Laplacian_c3,
    Laplacian_c4,
    Laplacian_Lux,
    Laplacian_Lux_pd,
    Laplacian_stencil,
    Laplacian_conv
]
for f in function_list
    println(f)
    f(grid_GS_u.grid_data, grid_GS_u.dx, grid_GS_u.dy)
end

# Check if they produce the same output
results = [method(grid_GS_u.grid_data, grid_GS_u.dx, grid_GS_u.dy)
           for method in function_list];
if all(results[1] ≈ result for result in results[2:end])
    println("All methods produce the same results")
else
    println("ERROR: Some methods produce different results")
end

# and test them by running them 10 times
using Statistics
results = Dict()
GC.gc()
for f in function_list
    times = Float64[]
    allocations = Int64[]
    gctime = Float64[]
    for i in 1:10
        result = @timed f(grid_GS_u.grid_data, grid_GS_u.dx, grid_GS_u.dy)
        push!(times, result[2])
        push!(allocations, result[3])
        push!(gctime, result[4])
    end
    GC.gc()
    results[string(f)] = (sum(times), sum(allocations), sum(gctime))
end

# sort the results by time
methods = []
times = []
allocations = []
gctimes = []
println("Results sorted by time")
for key in sort(collect(keys(results)), by = x -> results[x][1])
    push!(methods, key)
    push!(times, results[key][1])
    push!(allocations, results[key][2])
    push!(gctimes, results[key][3])
end

# And plot them
using Plots
p1 = bar(methods, times, title = "Execution Times", ylabel = "Time (s)",
    yaxis = :log, legend = false, xrotation = 45)
p2 = bar(methods, allocations, title = "Memory Allocations",
    xlabel = "Methods", ylabel = "Bytes", legend = false, xrotation = 45)
p3 = bar(methods, gctimes, title = "GC Time", xlabel = "Methods",
    ylabel = "Time (s)", legend = false, xrotation = 45)

P = plot(p1,
    p2,
    p3,
    layout = (3, 1),
    size = (600, 900),
    suptitle = "gridsize = $(size(u_initial)[1])x$(size(u_initial)[2])x$(size(u_initial)[3])")
savefig(P,
    "examples/plots/derivatives_CPUbenchmark_$(size(u_initial)[1])_$(size(u_initial)[2])_$(size(u_initial)[3]).png")

##############################
##############################
########## GPU  ##############
##############################
##############################
# If you are on a GPU, you can also test the GPU version of the Laplacian
using CUDA
CUDA.allowscalar(false)
using LuxCUDA
CUDA.reclaim()
CUDA.memory_status()

# Create the GPU grid
cugu = make_grid(dim = 2, dtype = MY_TYPE, dx = cu(dux), nx = cu(nux), dy = cu(duy),
    ny = cu(nuy), nsim = nsim, grid_data = cu(u_initial))
cugv = make_grid(dim = 2, dtype = MY_TYPE, dx = cu(dvx), nx = cu(nvx), dy = cu(dvy),
    ny = cu(nvy), nsim = cu(nsim), grid_data = cu(v_initial))
CUDA.memory_status()

# we need to port the functions in GPU things
upx0 = cu(similar(u_initial));
umx0 = cu(similar(u_initial));
upy0 = cu(similar(u_initial));
umy0 = cu(similar(u_initial));
uu = cu(zeros(MY_TYPE, size(u_initial)[1] + 2, size(u_initial)[2] + 2, size(u_initial)[3]));
uu2 = cu(zeros(
    MY_TYPE, size(u_initial)[1] + 2, size(u_initial)[2] + 2, 1, size(u_initial)[3]));
K = cu(K)
D_3D = cu(D_3D)
M = Lux.Chain(
    # Pad for PBC
    x -> begin
        @views uu2[2:(end - 1), 2:(end - 1), 1, :] .= x
        @views uu2[1, 2:(end - 1), 1, :] .= x[end, :, :]
        @views uu2[end, 2:(end - 1), 1, :] .= x[1, :, :]
        @views uu2[2:(end - 1), 1, 1, :] .= x[:, end, :]
        @views uu2[2:(end - 1), end, 1, :] .= x[:, 1, :]
        uu2
    end,
    Lux.Conv((3, 3), 1 => 1, use_bias = false),
    SelectDim(3, 1)
)
θ, st = setup(rng, M);
θ = ComponentArrays.ComponentArray(θ)
θ.layer_2.weight = [0.0f0 1.0f0 0.0f0;
                    1.0f0 -4.0f0 1.0f0;
                    0.0f0 1.0f0 0.0f0]
function Laplacian_Lux(u, Δx2, Δy2 = 0.0, Δz2 = 0.0)
    Lux.apply(M, u, θ, st)[1]
end
# this is a Lux variant with padding instead of preallocation
Mpd = Lux.Chain(
    # Pad for PBC
    WrappedFunction(x -> unsqueeze(NNlib.pad_circular(x, 1; dims = [1, 2]), 3)),
    #x -> reshape(x, size(x)[1], size(x)[2], 1, size(x)[3]),
    Lux.Conv((3, 3), 1 => 1, use_bias = false),
    SelectDim(3, 1)
)
θpd, stpd = Lux.setup(rng, Mpd);
θpd = ComponentArrays.ComponentArray(θ)
θpd.layer_2.weight = [0.0f0 1.0f0 0.0f0;
                      1.0f0 -4.0f0 1.0f0;
                      0.0f0 1.0f0 0.0f0]
function Laplacian_Lux_pd(u, Δx2, Δy2 = 0.0, Δz2 = 0.0)
    Lux.apply(Mpd, u, θpd, stpd)[1]
end
function Laplacian_stencil(u, Δx2, Δy2 = 0.0, Δz2 = 0.0)
    NNlib.batched_mul(K, u_initial) + NNlib.batched_mul(u_initial, K)
end
function Laplacian_conv(u, Δx2, Δy2 = 0.0, Δz2 = 0.0)
    # Create the ghost grid
    uu[2:(end - 1), 2:(end - 1), :] .= u
    uu[1, 2:(end - 1), :] .= u[end, :, :]
    uu[end, 2:(end - 1), :] .= u[1, :, :]
    uu[2:(end - 1), 1, :] .= u[:, end, :]
    uu[2:(end - 1), end, :] .= u[:, 1, :]
    dropdims(NNlib.conv(unsqueeze(uu, 3), D_3D), dims = 3)
end

Laplacian_Lux(cugu.grid_data, cugu.dx, cugu.dy);
Laplacian_Lux_pd(cugu.grid_data, cugu.dx, cugu.dy);
Old_Laplacian(cugu.grid_data, cugu.dx, cugu.dy);

# trigger the GPU version of the Laplacian
f_good = []
for f in function_list
    println("\n")
    println(f)
    try
        f(cugu.grid_data, cugu.dx, cugu.dy)
        push!(f_good, f)
        GC.gc()
        CUDA.reclaim()
        CUDA.memory_status()
    catch e
        println("Above method produced and error")
        #println(e)
    end
end
GC.gc()
CUDA.reclaim()
CUDA.memory_status()

printlnt("They compiled:")
println(f_good)

# and test them by running them 10 times
for f in f_good
    println("\n")
    println(f)
    @time for i in 1:10
        f(cugu.grid_data, cugu.dx, cugu.dy)
    end
    GC.gc()
    CUDA.reclaim()
    CUDA.memory_status()
end

# check the results
results = [method(grid_GS_u.grid_data, grid_GS_u.dx, grid_GS_u.dy) for method in f_good];
all(results[1] ≈ result for result in results[2:end])
