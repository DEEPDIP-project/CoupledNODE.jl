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
using Images
using Interpolations
using NNlib
ArrayType = CUDA.functional() ? CuArray : Array;
## Import our custom backend functions
include("coupling_functions/functions_example.jl")
include("coupling_functions/functions_NODE.jl")
include("coupling_functions/functions_loss.jl")
include("coupling_functions/functions_FDderivatives.jl");


#THis contains the final part for the grid discretization to use as example 2.2
# (only to show grid coarsening)






# ## Learning the Gray-Scott model

# We have seen in the following example how to use the CNODE to solve the Gray-Scott model.
# Here we want to introduce the Learning part of the CNODE, and show how it can be used to close the CNODE.

# As a reminder, the GS model is defined from 
# \begin{equation}\begin{cases} \frac{du}{dt} = D_u \nabla u - uv^2 + f(1-u)  \equiv F_u(u,v) \\ \frac{dv}{dt} = D_v \nabla v + uv^2 - (f+k)v  \equiv G_v(u,v)\end{cases} \end{equation}
# where $u(x,y,t):\mathbb{R}^2\times \mathbb{R}\rightarrow \mathbb{R}$ is the concentration of species 1, while $v(x,y,t)$ is the concentration of species two. This model reproduce the effect of the two species diffusing in their environment, and reacting together.
# This effect is captured by the ratios between $D_u$ and $D_v$ (the diffusion coefficients) and $f$ and $k$ (the reaction rates).

# Let's use the exact GS model to gather some data

# In following examples we will use the GS model to discuss the effect of numerically approximating PDEs using finite differences. 
# We will also focus on the showcasing the strenght of trained CNODEs.

# The system that we want to solve is the Gray-Scott model

# First we create a grid to discretize the problem. Notice that in literature the coefficients are usually scaled such that $dx=dy=1$, so we will use this scaling to have a direct comparison with literature.
dux = duy = dvx = dvy = 1.
nux = nuy = nvx = nvy = 100
grid = Grid(dux, duy, nux, nuy, dvx, dvy, nvx, nvy);


# We start with a central concentration of $v$
function initialize_uv(grid, u_bkg, v_bkg, center_size)
    u_initial = u_bkg*ones(grid.nux,grid.nuy)
    v_initial = 0*ones(grid.nvx,grid.nvy)
    v_initial[Int(grid.nvx/2-center_size):Int(grid.nvx/2+center_size),Int(grid.nvy/2-center_size):Int(grid.nvy/2+center_size)] .= v_bkg
    return u_initial, v_initial
end
u_initial, v_initial = initialize_uv(grid, 0.8, 0.9, 4);

# We can now define the initial condition as a flattened concatenated array
uv0 = vcat(reshape(u_initial, grid.nux*grid.nuy,1),reshape(v_initial, grid.nvx*grid.nvy,1));


# From the literature, we have selected the following parameters in order to form nice patterns
D_u = 0.16
D_v = 0.08
f = 0.055
k = 0.062;


# Here we (the user) define the **right hand sides** of the equations
F_u(u,v,grid) = D_u*Laplacian(u,grid.dux,grid.duy) .- u.*v.^2 .+ f.*(1.0.-u)
G_v(u,v,grid) = D_v*Laplacian(v,grid.dvx,grid.dvy) .+ u.*v.^2 .- (f+k).*v

# Once the forces have been defined, we can create the CNODE
# (Notice that atm we are not closing with the NNN, so it is not requried yet) [TO DO]
f_CNODE = create_f_CNODE(F_u, G_v, grid; is_closed=false);
# and we ask Lux for the parameters to train and their structure [useless in this case]
θ, st = Lux.setup(rng, f_CNODE);

# We now do a short *burnout run* to get rid of the initial artifacts
trange_burn = (0.0f0, 10.0f0)
dt, saveat = (1e-2, 1)
full_CNODE = NeuralODE(f_CNODE, trange_burn, Tsit5(), adaptive=false, dt=dt, saveat=saveat);
burnout_CNODE_solution = Array(full_CNODE(uv0, θ, st)[1]);


# **CNODE run** 
# We use the output of the burnout to start a longer simulations
uv0 = burnout_CNODE_solution[:,:,end];
trange = (0.0f0, 8000.0f0)
dt, saveat = (1/(4*max(D_u,D_v)), 25)
full_CNODE = NeuralODE(f_CNODE, trange, Tsit5(), adaptive=false, dt=dt, saveat=saveat);
untrained_CNODE_solution = Array(full_CNODE(uv0, θ, st)[1])
# And we unpack the solution to get the two species from
u = reshape(untrained_CNODE_solution[1:grid.Nu, :, :]    , grid.nux, grid.nuy, size(untrained_CNODE_solution,2), :);
v = reshape(untrained_CNODE_solution[grid.Nu+1:end, :, :], grid.nvx, grid.nvy, size(untrained_CNODE_solution,2), :);


# Plot the solution as an animation
anim = Animation()
fig = plot(layout = (1, 2), size = (600, 300))
@gif for i in 1:2:size(u, 4)
    p1 = heatmap(u[:,:,1,i], axis=false, bar=true, aspect_ratio=1, color=:reds , title="u(x,y)")
    p2 = heatmap(v[:,:,1,i], axis=false, bar=true, aspect_ratio=1, color=:blues, title="v(x,y)")
    time = round(i*saveat, digits=0)
    fig = plot(p1, p2, layout=(1,2), plot_title="time = $(time)")
    frame(anim, fig)
end
if isdir("./plots")
    gif(anim, "./plots/GS.gif", fps=10)
else
    gif(anim, "examples/plots/GS.gif", fps=10)
end


# ### Errors induced by coarsening
# We show now what happens if instead of the standard discretization `dx=dy=1`, we use a coarser one on the second species.
# So now we redefine the grid parameters
nvx = nvy = 75
dvx = nux*dux/nvx
dvy = nuy*duy/nvy
coarse_grid = Grid(dux, duy, nux, nuy, dvx, dvy, nvx, nvy);

# Once we have non matching grids, the forces have to match this new structure. 
# To do so, we have designed the CNODE to perform a differentiable interpolation step to match the two species on the same grid.

# So we can create the CNODE, that will automatically know if the interpolation is needed
f_coarse_CNODE = create_f_CNODE(F_u, G_v, coarse_grid; is_closed=false)
θ, st = Lux.setup(rng, f_coarse_CNODE);

# For the initial condition, we take the finer grid after the burnout and filter the v component to the coarse grid. 
uv0 = burnout_CNODE_solution[:,:,end];
u0b = reshape(uv0[1:grid.Nu, 1]    , grid.nux, grid.nuy);
v0b = reshape(uv0[grid.Nu+1:end, 1], grid.nvx, grid.nvy, );
v0_coarse = imresize(v0b, (coarse_grid.nvx,coarse_grid.nvy),method=Lanczos4OpenCV());
uv0_coarse = vcat(reshape(u0b, coarse_grid.Nu,:),reshape(v0_coarse, coarse_grid.Nv,:));

# **CNODE run** 
coarse_CNODE = NeuralODE(f_coarse_CNODE, trange, Tsit5(), adaptive=false, dt=dt, saveat=saveat)
coarse_CNODE_solution = Array(coarse_CNODE(uv0_coarse, θ, st)[1]);
u_coarse = reshape(coarse_CNODE_solution[1:coarse_grid.Nu, :, :]    , coarse_grid.nux, coarse_grid.nuy, size(coarse_CNODE_solution,2), :);
v_coarse = reshape(coarse_CNODE_solution[coarse_grid.Nu+1:end, :, :], coarse_grid.nvx, coarse_grid.nvy, size(coarse_CNODE_solution,2), :);


# Compare the fine-fine solution with the fine-coarse solution
anim = Animation()
fig = plot(layout = (2, 2), size = (500, 500))
@gif for i in 1:1:size(u, 4)
    p1 = heatmap(u[:,:,1,i], axis=false, cbar=false, aspect_ratio=1, color=:reds ) 
    p2 = heatmap(v[:,:,1,i], axis=false, cbar=false, aspect_ratio=1, color=:blues)
    p3 = heatmap(u_coarse[:,:,1,i],axis=false, cbar=false, aspect_ratio=1, color=:reds ) 
    p4 = heatmap(v_coarse[:,:,1,i],axis=false, cbar=false, aspect_ratio=1, color=:blues)
    time = round(i*saveat, digits=2)
    fig = plot(p1, p2, p3, p4, layout=(2,2), title=["Fine Grid" "" "Coarse Grid (only for v)" ""], title_location=:center)
    frame(anim, fig)
end
if isdir("./plots")
    gif(anim, "plots/GS_coarse.gif", fps=10)
else
    gif(anim, "examples/plots/GS_coarse.gif", fps=10)
end

# From the figure, you can see that the coarser discretization of v has induced some artifacts that influences the whole dynamics. We will solve these artifacts using the Neural part of the CNODES.

# In the second part of this example we will show how to use the Neural Network to close the CNODE.