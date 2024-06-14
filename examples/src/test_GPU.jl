using Lux
using LuxCUDA
using SciMLSensitivity
using DiffEqFlux
using DifferentialEquations
#using Plots
#using Plots.PlotMeasures
using Zygote
using Random
rng = Random.seed!(1234)
#using OptimizationOptimisers
#using Statistics
using ComponentArrays
using CUDA
#using Images
#using Interpolations
#using NNlib
#using FFTW
using DiffEqGPU
# Test if CUDA is running
CUDA.functional()
CUDA.allowscalar(false)
const ArrayType = CUDA.functional() ? CuArray : Array;
const z = CUDA.functional() ? CUDA.zeros : (s...) -> zeros(Float32, s...);
const solver_algo = CUDA.functional() ? GPUTsit5() : Tsit5();
const MY_DEVICE = CUDA.functional() ? cu : identity;
# and remember to use float32 if you plan to use a GPU
const MY_TYPE = Float32;

# ## Testing SciML on GPU

# In this example, we want to test the GPU implementation of SciML. We also use float32 to speed up the computation on the GPU. 
## on snellius test nodes, this is the largest grid that fits on the GPU
## [notice that for smaller grids it is not convenient to use the GPU at all]
#nux = nuy = nvx = nvy = 50;
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

using ShiftedArrays
include("./../../src/derivatives.jl")
function F_u(u, v)
    D_u * Laplacian(u, grid_GS_u.dx, grid_GS_u.dy) .- u .* v .^ 2 .+
    f .* (1.0f0 .- u)
end

using Profile
# Trigger and test if the force keeps the array on the GPU
@time zz = F_u(u_initial, v_initial);
cuu = cu(u_initial);
cuv = cu(v_initial);
@time zz = F_u(cuu, cuv);
@time for i in 1:1000
    ##@profview for i in 1:100
    ##CUDA.@profile for i in 1:100
    zz = F_u(u_initial, v_initial)
end
@time for i in 1:1000
    ##CUDA.@profile for i in 1:100
    ##@profview for i in 1:100
    zz = F_u(cuu, cuv)
end

# Typical cnode
f_CNODE_cpu = create_f_CNODE((F_u, G_v), (grid_GS_u, grid_GS_v); is_closed = false)
θ_cpu, st_cpu = Lux.setup(rng, f_CNODE_cpu);
# the only difference for the gpu is that the grid lives on the gpu now
#f_CNODE_gpu = create_f_CNODE(F_u, G_v, cu(grid); is_closed = false);
f_CNODE_gpu = create_f_CNODE(
    create_functions, D_u, D_v, f, k, cu(grid); is_closed = false, gpu_mode = true);
f_CNODE_gpu = create_f_CNODE((F_u, G_v), (grid_GS_u, grid_GS_v); is_closed = false)
#f_CNODE_gpu = f_CNODE_gpu |> gpu;
θ_gpu, st_gpu = Lux.setup(rng, f_CNODE_gpu);
#θ_gpu = θ_gpu |> gpu;
#st_gpu = st_gpu |> gpu;

# Trigger and test if the right hand side of the CNODE keeps input on the GPU
@time for i in 1:1000
    zz = f_CNODE_cpu(uv0, θ_gpu, st_gpu)
end
#@time zz = f_CNODE_gpu(uv0, θ_gpu, st_gpu);
cuv0 = cu(uv0);
@time for i in 1:1000
    zz = f_CNODE_gpu(cuv0, θ_gpu, st_gpu)
end
zz = nothing
GC.gc()

# [!] It is important that trange, dt, saveat have the correct type
trange_burn = (0.0f0, 20.0f0);
dt, saveat = (0.01f0, 1.0f0);
# [!] According to https://docs.sciml.ai/DiffEqGPU/stable/getting_started/ 
#     the best thing to do in case of bottleneck consisting in expensive right hand side is to use CuArray as initial condition and do not rely on EnsemblesGPU.

# But since this is the first time example where we use a GPU, let's do a CPU/GPU comparison 
full_CNODE_cpu = NeuralODE(f_CNODE_cpu,
    trange_burn,
    Tsit5(),
    adaptive = false,
    dt = dt,
    saveat = saveat);
full_CNODE_gpu = cu(NeuralODE(f_CNODE_gpu,
    trange_burn,
    GPUTsit5(),
    adaptive = false,
    dt = dt,
    saveat = saveat));
# also test with the initial condition on gpu
cu0 = cu(uv0);
typeof(uv0)
typeof(cu0)

# Now test the cpu_vs_gpu algorithms and the cpu_vs_gpu initial condition 
@time cpu_cpu = full_CNODE_cpu(uv0, θ_cpu, st_cpu)[1];
typeof(cpu_cpu.u)
# Clean the GPU memory and trigger gc
cpu_cpu = cpu_gpu = gpu_cpu = gpu_cpu = nothing
GC.gc()

#@time cpu_gpu = full_CNODE_cpu(cu0, θ_cpu, st_cpu)[1]; 
#typeof(cpu_gpu.u)
#cpu_cpu = cpu_gpu = gpu_cpu = gpu_cpu = nothing
#GC.gc()

CUDA.reclaim()
CUDA.memory_status()
CUDA.unsafe_free!(cu0)

@time gpu_gpu = full_CNODE_gpu(cu0, θ_gpu, st_gpu)[1];
typeof(gpu_gpu.u)
cpu_cpu = cpu_gpu = gpu_cpu = gpu_cpu = nothing
GC.gc(true)
