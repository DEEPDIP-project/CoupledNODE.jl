using Lux
using SciMLSensitivity
using DiffEqFlux
using DifferentialEquations
using Plots
using Zygote
using Random
rng = Random.seed!(1234)
using OptimizationOptimisers
using Statistics
using ComponentArrays
using CUDA
using BlockDiagonals
using Images
using Interpolations
ArrayType = CUDA.functional() ? CuArray : Array

# Import our custom backend functions
include("coupling_functions/functions_example.jl")
include("coupling_functions/functions_NODE.jl")
include("coupling_functions/functions_loss.jl")
include("coupling_functions/functions_FDderivatives.jl")


# We want to solve the Gray-Scott model
# u_t = D_u Δu - uv^2 + f(1-u)  = F_u(u,v)
# v_t = D_v Δv + uv^2 - (f+k)v  = G_v(u,v)
# where u and v are the concentrations of two chemical species, D_u and D_v are the diffusion coefficients, and f and k are the reaction rates.

# In following examples we will discuss the effect of numerically approximating the Laplacian operator Δu and Δv using finite differences. Here instead, we will just focus on obtaining a solution trough CNODEs.

# create the grid (this problem is scaled such that dx=dy=1) in this way we can refer to literature
dux = duy = dvx = dvy = 1
nux = nuy = nvx = nvy = 100
grid = Grid(dux, duy, nux, nuy, dvx, dvy, nvx, nvy)

## The initial condition is equal concentration wih random perturbation
#function initial_condition(grid, U₀, V₀, ε_u, ε_v)
#    u_init = U₀ .+ ε_u .* randn(grid.nux, grid.nuy)
#    v_init = V₀ .+ ε_v .* randn(grid.nvx, grid.nvy)
#    return u_init, v_init
#end
#U₀ = 0.5    # initial concentration of u
#V₀ = 0.25   # initial concentration of v
#ε_u = 0.05 # magnitude of the perturbation on u
#ε_v = 0.05 # magnitude of the perturbation on v
#u_initial, v_initial = initial_condition(grid, U₀, V₀, ε_u, ε_v)

# Start from a central concentration of v
function initialize_uv(grid, u_bkg, v_bkg, center_size)
    u_initial = u_bkg*ones(grid.nux,grid.nuy)
    v_initial = 0*ones(grid.nvx,grid.nvy)
    v_initial[Int(grid.nvx/2-center_size):Int(grid.nvx/2+center_size),Int(grid.nvy/2-center_size):Int(grid.nvy/2+center_size)] .= v_bkg
    return u_initial, v_initial
end
u_initial, v_initial = initialize_uv(grid, 0.8, 0.9, 4)
uv0 = BlockDiagonal([u_initial, v_initial])

# set the diffusion coefficients and the reaction rates
D_u = 0.16
D_v = 0.08
f = 0.055
k = 0.062


# Now the user defines the right hand sides of the equations
F_u(u,v,grid) = D_u*Laplacian(u,grid.dux,grid.duy) .- u.*v.^2 .+ f.*(1.0.-u)
G_v(u,v,grid) = D_v*Laplacian(v,grid.dvx,grid.dvy) .+ u.*v.^2 .- (f+k).*v

f_CNODE = create_f_CNODE(F_u, G_v, grid; is_closed=false)
# and get the parametrs that you want to train
θ, st = Lux.setup(rng, f_CNODE)

# * We define the CNODE (burnout)
trange_burn = (0.0f0, 10.0f0)
dt, saveat = (1e-2, 1)
full_CNODE = NeuralODE(f_CNODE, trange_burn, Tsit5(), adaptive=false, dt=dt, saveat=saveat)
# we also solve it, using the zero-initialized parameters
burnout_CNODE_solution = Array(full_CNODE(uv0, θ, st)[1])


# *** CNODE run 
uv0 = burnout_CNODE_solution[:,:,end]
trange = (0.0f0, 8000.0f0)
dt, saveat = (1/(4*max(D_u,D_v)), 25)
full_CNODE = NeuralODE(f_CNODE, trange, Tsit5(), adaptive=false, dt=dt, saveat=saveat)
# we also solve it, using the zero-initialized parameters
untrained_CNODE_solution = Array(full_CNODE(uv0, θ, st)[1])
u = untrained_CNODE_solution[1:end÷2,1:end÷2, :]
v = untrained_CNODE_solution[end÷2+1:end,end÷2+1:end, :]


# plot the solution as an animation
anim = Animation()
fig = plot(layout = (1, 2), size = (600, 300))
@gif for i in 1:size(u, 3)
    p1 = heatmap(u[:,:,i], axis=false, bar=true, aspect_ratio=1, color=:reds , title="u") 
    p2 = heatmap(v[:,:,i], axis=false, bar=true, aspect_ratio=1, color=:blues, title="v")
    time = round(i*saveat, digits=2)
    fig = plot(p1, p2, layout=(1,2), title="time = $(time)")
    frame(anim, fig)
end
gif(anim, "plots/GS.gif", fps=10)


# ****** Now we will compare the solution with a coarser grid (coarser only on v)

# Now filter v to get a 50x50 grid, each element is the average of 4 elements
nvx = nvy = 95
dvx = 100.0/nvx
dvy = 100.0/nvy
coarse_grid = Grid(dux, duy, nux, nuy, dvx, dvy, nvx, nvy)


# Now the user defines the right hand sides of the equations
Fc_u(u,v, grid) = begin
    v_resized = imresize(v, (grid.nux,grid.nuy), method=Lanczos4OpenCV())
    F_u(u,v_resized, grid)
end
Gc_v(u,v, grid) = begin
    u_resized = imresize(u, (grid.nvx,grid.nvy), method=Lanczos4OpenCV())
    G_v(u_resized,v, grid)
end

f_coarse_CNODE = create_f_CNODE(Fc_u, Gc_v, coarse_grid; is_closed=false)
# and get the parametrs that you want to train
θ, st = Lux.setup(rng, f_coarse_CNODE)


# Take the full grid after the burnout and filter to get the initial condition
uv0 = burnout_CNODE_solution[:,:,end]
u0b = uv0[1:end÷2,1:end÷2]
v0b = uv0[end÷2+1:end,end÷2+1:end]
v0_coarse = imresize(v0b, (coarse_grid.nvx,coarse_grid.nvy))
uv0_coarse = BlockDiagonal([u0b, v0_coarse])

# *** CNODE run 
coarse_CNODE = NeuralODE(f_coarse_CNODE, trange, Tsit5(), adaptive=false, dt=dt, saveat=saveat)
# we also solve it, using the zero-initialized parameters
coarse_CNODE_solution = Array(coarse_CNODE(uv0_coarse, θ, st)[1])
u_coarse = coarse_CNODE_solution[1:coarse_grid.nux,1:coarse_grid.nuy, :]
v_coarse = coarse_CNODE_solution[coarse_grid.nux+1:end,coarse_grid.nuy+1:end, :]


# plot the solution as an animation comparing the two resolutions
anim = Animation()
fig = plot(layout = (2, 2), size = (500, 500))
@gif for i in 1:size(u, 3)
    p1 = heatmap(u[:,:,i], axis=false, cbar=false, aspect_ratio=1, color=:reds ) 
    p2 = heatmap(v[:,:,i], axis=false, cbar=false, aspect_ratio=1, color=:blues)
    p3 = heatmap(u_coarse[:,:,i],axis=false, cbar=false, aspect_ratio=1, color=:reds ) 
    p4 = heatmap(v_coarse[:,:,i],axis=false, cbar=false, aspect_ratio=1, color=:blues)
    time = round(i*saveat, digits=2)
    fig = plot(p1, p2, p3, p4, layout=(2,2))
    frame(anim, fig)
end
gif(anim, "plots/GS_coarse.gif", fps=10)