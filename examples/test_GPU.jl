using Lux
using LuxCUDA
using SciMLSensitivity
using DiffEqFlux
using DifferentialEquations
using Plots
using Plots.PlotMeasures
using Zygote
using Random
rng = Random.seed!(1234)
using OptimizationOptimisers
using Statistics
using ComponentArrays
using CUDA
using Images
using Interpolations
using NNlib
using FFTW
using DiffEqGPU
# Test if CUDA is running
CUDA.functional()
CUDA.allowscalar(false)
const ArrayType = CUDA.functional() ? CuArray : Array;
const z = CUDA.functional() ? CUDA.zeros : (s...) -> zeros(Float32, s...);
const solver_algo = CUDA.functional() ? GPUTsit5() : Tsit5();
# and remember to use float32 if you plan to use a GPU
const MY_TYPE = Float32;
## Import our custom backend functions
include("coupling_functions/functions_example.jl")
include("coupling_functions/functions_NODE.jl")
include("coupling_functions/functions_CNODE_loss.jl")
include("coupling_functions/functions_FDderivatives.jl");
include("coupling_functions/functions_nn.jl")
include("coupling_functions/functions_FNO.jl")

# ## Testing SciML on GPU

# In this example, we want to test the GPU implementation of SciML. We also use float32 to speed up the computation on the GPU. 

dux = duy = dvx = dvy = 1.0f0;
nux = nuy = nvx = nvy = 40;
grid = Grid(dux, duy, nux, nuy, dvx, dvy, nvx, nvy, convert_to_float32 = true);
function initial_condition(grid, U₀, V₀, ε_u, ε_v; nsimulations = 1)
    u_init = MY_TYPE.(U₀ .+ ε_u .* randn(grid.nux, grid.nuy, nsimulations))
    v_init = MY_TYPE.(V₀ .+ ε_v .* randn(grid.nvx, grid.nvy, nsimulations))
    return u_init, v_init
end
U₀  = 0.5f0;
V₀  = 0.25f0;
ε_u = 0.05f0;
ε_v = 0.1f0;
u_initial, v_initial = initial_condition(grid, U₀, V₀, ε_u, ε_v, nsimulations = 4);

# We can now define the initial condition as a flattened concatenated array
uv0 = MY_TYPE.(vcat(reshape(u_initial, grid.Nu, :), reshape(v_initial, grid.nvx * grid.nvy, :)));

D_u = 0.16f0;
D_v = 0.08f0;
f = 0.055f0;
k = 0.062f0;

# RHS of GS model
F_u(u, v, grid) = MY_TYPE.(D_u * Laplacian(u, grid.dux, grid.duy) .- u .* v .^ 2 .+ f .* (1.0f0 .- u));
G_v(u, v, grid) = MY_TYPE.(D_v * Laplacian(v, grid.dvx, grid.dvy) .+ u .* v .^ 2 .- (f + k) .* v);
# Typical cnode
f_CNODE_cpu = create_f_CNODE(F_u, G_v, grid; is_closed = false);
θ_cpu, st_cpu = Lux.setup(rng, f_CNODE_cpu);
# the only difference for the gpu is that the grid lives on the gpu now
f_CNODE_gpu = create_f_CNODE(F_u, G_v, cu(grid); is_closed = false);
θ_gpu, st_gpu = Lux.setup(rng, f_CNODE_gpu);


# Test if the force keeps the array on the GPU
zz = F_u(u_initial, v_initial, grid)
typeof(zz)
zz = F_u(cu(u_initial), cu(v_initial), cu(grid))
typeof(zz)
# Test if the right hand side of the CNODE is on the GPU
zz = f_CNODE_gpu(uv0, θ_gpu, st_gpu)
typeof(zz)
zz = f_CNODE_gpu(cu(uv0), θ_gpu, st_gpu)
typeof(zz)
zz = nothing
GC.gc()


trange_burn = (0.0f0, 0.10f0);
dt, saveat = (1e-2, 1);
# [!] According to https://docs.sciml.ai/DiffEqGPU/stable/getting_started/ 
#     the best thing to do in case of bottleneck consisting in expensive right hand side is to use CuArray as initial condition and do not rely on EnsemblesGPU.

# But since this is the first time example where we use a GPU, let's do a CPU/GPU comparison 
include("coupling_functions/functions_NODE.jl")
full_CNODE_cpu = NeuralODE(f_CNODE_cpu,
    trange_burn,
    Tsit5(),
    adaptive = false,
    dt = dt,
    saveat = saveat);
full_CNODE_gpu = NeuralODE(f_CNODE_gpu,
    trange_burn,
    GPUTsit5(),
    adaptive = false,
    dt = dt,
    saveat = saveat);
# also test with the initial condition on gpu
cu0 = cu(uv0);
typeof(uv0)
typeof(cu0)

# Now test the cpu_vs_gpu algorithms and the cpu_vs_gpu initial condition 
@time cpu_cpu = full_CNODE_cpu(uv0, θ_cpu, st_cpu)[1];
# on which device is the result?
typeof(cpu_cpu)
# Clean the GPU memory and trigger gc
cpu_cpu = cpu_gpu = gpu_cpu = gpu_cpu = nothing
GC.gc()
# @time cpu_gpu = full_CNODE_cpu(cu0, θ, st)[1]; # not woth, too slow
@time gpu_cpu = full_CNODE_gpu(uv0, θ_gpu, st_gpu)[1];
typeof(gpu_cpu)
cpu_cpu = cpu_gpu = gpu_cpu = gpu_cpu = nothing
GC.gc()
@time gpu_gpu = full_CNODE_gpu(cu0, θ_gpu, st_gpu)[1];
typeof(gpu_gpu)
cpu_cpu = cpu_gpu = gpu_cpu = gpu_cpu = nothing
GC.gc()

