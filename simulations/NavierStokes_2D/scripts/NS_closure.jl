using CairoMakie
using IncompressibleNavierStokes
# [!] you can not use the abbreviation 'INS' with Enzyme, because it does not know how to differentiate the getname function!
INS = IncompressibleNavierStokes

# [https://github.com/JuliaMath/AbstractFFTs.jl/issues/99] we cannot define reverse-mode rules for mul!(y::AbstractArray, plan::Plan, x::AbstractArray), since we don't know what the plan is and therefore don't know how to normalize.


# Setup and initial condition
T = Float32
ArrayType = Array
Re = T(1_000)
n = 256
lims = T(0), T(1)
x , y = LinRange(lims..., n + 1), LinRange(lims..., n + 1)
setup = Setup(x, y; Re, ArrayType);
ustart = random_field(setup, T(0));
psolver = psolver_spectral(setup);
dt = T(1e-3)
trange = (T(0), T(1))
savevery = 20
saveat = 20 * dt
#trange = (T(0), T(10*dt))
#savevery = 1
#saveat = dt

# Syver's solver 
# we know that mathematically it is the best one 
(state, outputs), time_ins, allocation, gc, memory_counters = @timed solve_unsteady(;
    setup,
    ustart,
    tlims = trange,
    Δt = dt,
    processors = (
        ehist = realtimeplotter(;
            setup,
            plot = energy_history_plot,
            nupdate = 10,
            displayfig = false
        ),
        anim = animator(; setup, path = "./vorticity.mkv", nupdate = savevery),
        #espec = realtimeplotter(; setup, plot = energy_spectrum_plot, nupdate = 10),
        log = timelogger(; nupdate = 100)
    )
);


############# Using SciML
using DifferentialEquations

# Projected force for SciML, to use in CNODE
F = similar(stack(ustart))
cache_F = (F[:,:,1], F[:,:,2])
cache_div = INS.divergence(ustart,setup)
cache_p = INS.pressure(ustart, nothing, 0.0f0, setup; psolver)
cache_out = similar(F)
Ω = setup.grid.Ω

# [!] I have marked the poisson solver as inactive
using Enzyme
EnzymeRules.inactive(::typeof(IncompressibleNavierStokes.poisson!), args...) = nothing

# * There is also an in-place version that takes the cache as a parameter p 
F_ip(du, u, p, t)= begin
    u_view = eachslice(u; dims = 3)
    IncompressibleNavierStokes.apply_bc_u!(u_view, t, setup)
    IncompressibleNavierStokes.momentum!(p[1], u_view, nothing, t, setup)
    IncompressibleNavierStokes.apply_bc_u!(p[1], t, setup; dudt = true)
    #########
    T = eltype(u_view[1])
    IncompressibleNavierStokes.divergence!(p[2], p[1], setup)
    @. p[2] *= Ω
    # Solve the Poisson equation
    IncompressibleNavierStokes.poisson!(p[4], p[3], p[2])
    IncompressibleNavierStokes.apply_bc_p!(p[3], T(0), setup)
    # Apply pressure correction term
    IncompressibleNavierStokes.applypressure!(p[1], p[3], setup)
    #########
    #IncompressibleNavierStokes.project!(p[1], setup; psolver=p[4], div=p[2], p=p[3])
    IncompressibleNavierStokes.apply_bc_u!(p[1], t, setup; dudt = true)
    du[:,:,1] .= p[1][1]
    du[:,:,2] .= p[1][2]
    nothing
end
temp = similar(stack(ustart))
myfull_cache = (cache_F,cache_div, cache_p, psolver)
F_ip(temp, stack(ustart), myfull_cache, 0.0f0);
using Plots
Plots.heatmap(myfull_cache[1][1] - ustart[1])

#struct Offset{D} end
#@inline (::Offset{D})(α) where {D} = CartesianIndex(ntuple(β -> β == α ? 1 : 0, D))




# Solve the ODE using ODEProblem
prob = ODEProblem{true}(F_ip, stack(ustart), trange, p=myfull_cache)
sol_ode, time_ode, allocation_ode, gc_ode, memory_counters_ode = @timed solve(
    prob,
    p = myfull_cache,
    RK4();
    dt = dt,
    saveat = saveat,
);

# I would like to train the ML using in-place updates that can maybe work with Enzyme. However the code below throws an error due to some mutation. 
# For this reason I am trying to use the `F_syv` function that does not mutate the input. However this requires a different NN since the in place operation works with tuples

import Random, Lux;
Random.seed!(123);
rng = Random.default_rng();

# Test 1:
# out-of-place update
dummy_NN = Lux.Chain(
#    u -> u.*3,
    Lux.Scale((1,1)),
)
θ_node, st_node = Lux.setup(rng, dummy_NN)
θ_node.weight .= 1
θ_node.bias.= 0
θ_node
using ComponentArrays
Lux.apply(dummy_NN, stack(ustart), ComponentArray(θ_node), st_node)[1]

# We do not need to use the Skipblock
#Fnode_op = Lux.Chain(
#    Lux.SkipConnection(
#        dummy_NN,
#        (f_NN, uv) -> begin
#            #F_syv(uv, nothing, 0) .+ f_NN
#            F_syv(uv, nothing, 0) #.+ f_NN
#        end; name = "Closure"),
#)
#Fnode_op = Lux.Chain(dummy_NN)
#θ_node, st_node = Lux.setup(rng, Fnode_op)
# dudt_out(u::Array{Float32}, θ::ComponentArray{Float32}, t::Float32) = Lux.apply(Fnode_op, u, θ, st_node)[1]

# Also we do not need the out-of-place version
#create_rhs_full_oop(setup, psolver) = function rhs(u::Array{Float32}, p::Any, t::Float32)
#    u = eachslice(u; dims = ndims(u))
#    u = (u...,)
#    u = IncompressibleNavierStokes.apply_bc_u(u, t, setup)
#    F = INS.momentum(u, nothing, t, setup)
#    F = INS.apply_bc_u(F, t, setup; dudt = true)
#    PF = INS.project(F, setup; psolver)
#    stack(PF)
#end
#F_syv = create_rhs_full_oop(setup, psolver)
#dudt_out(u::Array{Float32}, θ::ComponentArray{Float32}, t::Float32) = F_syv(u, nothing, t) .+ Lux.apply(dummy_NN, u, θ, st_node)[1]
#dudt_out(stack(ustart), ComponentArray(θ_node), 0.0f0)
#prob_node = ODEProblem{false}(dudt_out, stack(ustart), trange)

## This is the in-place version
#dudt_out(du::Array{Float32}, u::Array{Float32}, θ::ComponentArray{Float32}, t::Float32) = begin
#    F_ip(du, u, nothing, t) 
#    gg = Lux.apply(dummy_NN, u, θ, st_node)[1]
#    @. du .= du .+ gg
#end
#dudt_out(temp, stack(ustart), ComponentArray(θ_node), 0.0f0)
#prob_node = ODEProblem{true}(dudt_out, stack(ustart), trange)

# Alternative in-place version
dudt_out(du::Array{Float32}, u::Array{Float32}, P, t::Float32) = begin
    F_ip(du, u, P[1], t) 
    gg = Lux.apply(dummy_NN, u, P[2], st_node)[1]
    @. du .= du .+ gg
end
P1 = (cache_F,cache_div, cache_p, psolver)
dudt_out(temp, stack(ustart), (P1,ComponentArray(θ_node)), 0.0f0)
prob_node = ODEProblem{true}(dudt_out, stack(ustart), trange, p=(P1,ComponentArray(θ_node)))
prob_node = ODEProblem(dudt_out, stack(ustart), trange, (P1,ComponentArray(θ_node)))

u0stacked = stack(ustart)
sol_node, time_node, allocation_node, gc_node, memory_counters_node = @timed solve(prob_node, RK4(), u0 = u0stacked, p = (P1,ComponentArray(θ_node)), saveat = saveat, dt=dt);

### Test 2:
### in-place update 
### Alternatively, we can follow this example to get a node using ODEProblem and in-place updates
### https://docs.sciml.ai/SciMLSensitivity/stable/examples/neural_ode/neural_ode_flux/
##
### for in place updates, we need to capture the `du` in the Lux layer
### so there is this function to create the Fnode with `du` captured
##function make_Fnode(du, t)
##    Fnode = Lux.Chain(
##        Lux.SkipConnection(
##            dummy_NN,
##            (f_NN, uv) -> begin
##                F_ip(du, uv, nothing, t)
##                @. du .= du .+ f_NN
##            end; name = "Closure"),
##    )
###    return Fnode
##    return Lux.Chain(dummy_NN,
##    )
##end
##
### This alternative instead is 
### Adjust dudt to use the modified Fnode that captures `du`
##function dudt2(du, u, θ, t)
##    # Create Fnode with `du` captured
##    Fnode_with_du = make_Fnode(du, t)
##    θ_node, st_node = Lux.setup(rng, Fnode_with_du)
##    return Lux.apply(Fnode_with_du, u, θ, st_node)
##end
##
### Define a higher-order function to initialize Fnode_with_du and setup only once
##function create_dudt_function(du, t)
##    # Initialize Fnode_with_du and setup only once
##    Fnode_with_du = make_Fnode(du, t)
##    θ_node, st_node = Lux.setup(rng, Fnode_with_du)
##    
##    # Return a new function that captures the initialized state and only calls Lux.apply
##    return (du, u, θ, t) -> Lux.apply(Fnode_with_du, u, θ, st_node)
##end
##
### Example usage:
### Initialize dudt with the necessary state
##dudt = create_dudt_function(temp, 0)
### obtain a copy of the initial state
##Fnode_with_du = make_Fnode(temp, 0)
##θ_node, st_node = Lux.setup(rng, Fnode_with_du)
##
##dudt(temp, stack(ustart), θ_node, 0.0f0);
##
##
##
##prob_node = ODEProblem{true}(dudt, stack(ustart), trange)
##prob_node = ODEProblem{true}(dudt2, stack(ustart), trange)
##
##sol_node, time_node, allocation_node, gc_node, memory_counters_node = @timed solve(prob_node, RK4(), u0 = stack(ustart), p = θ_node, saveat = saveat, dt=dt);



# Compare the times of the different methods via a bar plot
using Plots
p1=Plots.bar(["INS", "ODE", "CNODE"], [time_ins, time_ode, time_node], xlabel = "Method", ylabel = "Time (s)", title = "Time comparison")
# Compare the memory allocation
p2=Plots.bar(["INS", "ODE", "CNODE"], [memory_counters.allocd, memory_counters_ode.allocd, memory_counters_node.allocd], xlabel = "Method", ylabel = "Memory (bytes)", title = "Memory comparison")
# Compare the number of garbage collections
p3=Plots.bar(["INS", "ODE", "CNODE"], [gc, gc_ode, gc_node], xlabel = "Method", ylabel = "Number of GC", title = "GC comparison")

Plots.plot(p1, p2, p3, layout=(3,1), size=(600, 800))



# Plot the final state
using Plots
p1=Plots.heatmap(title="u in SciML ODE",sol_ode.u[end][:, :, 1])
p2=Plots.heatmap(title="u in SciML CNODE",sol_node.u[end][:, :, 1])
p3=Plots.heatmap(title="u in INS",state.u[1])
# and compare them
p4=Plots.heatmap(title="u_INS-u_ODE",state.u[1] - sol_ode.u[end][:, :, 1])
p5=Plots.heatmap(title="u_INS-u_CNODE",state.u[1] - sol_node.u[end][:, :, 1])
p6=Plots.heatmap(title="u_CNODE-u_ODE",sol_node.u[end][:, :, 1] - sol_ode.u[end][:, :, 1])
Plots.plot(p1, p2, p3, p4,p5,p6, layout=(2,3), size=(900,600))

# and plot the vorticty as well
vins = INS.vorticity((state.u[1], state.u[2]), setup)
vode = INS.vorticity((sol_ode.u[end][:, :, 1], sol_ode.u[end][:, :, 2]), setup) 
vnode = INS.vorticity((sol_node.u[end][:, :, 1], sol_node.u[end][:, :, 2]), setup) 

Plots.heatmap(title="vorticity INS-ODE",  - )
p1=Plots.heatmap(title="vorticity in SciML ODE",vode)
p2=Plots.heatmap(title="vorticity in SciML CNODE",vnode)
p3=Plots.heatmap(title="vorticity in INS",vins)
# and compare them
p4=Plots.heatmap(title="INS-ODE",vins-vode)
p5=Plots.heatmap(title="INS-CNODE",vins-vnode)
p6=Plots.heatmap(title="CNODE-ODE",vode-vnode)
Plots.plot(p1, p2, p3, p4,p5,p6, layout=(2,3), size=(900,600))



### Animate solution using Makie
### ! we can plot either the vorticity or the velocity field, however notice that the vorticity is 'flashing' (unstable)
##using GLMakie
##let
##    (; Iu) = setup.grid
##    i = 1
###    obs = Observable(sol.u[1][Iu[i], i])
###    ω = IncompressibleNavierStokes.vorticity((u[:, :, 1], u[:, :, 2]), setup)
##    obs = Observable(INS.vorticity((sol_ode.u[1][:, :, 1], sol_ode.u[1][:, :, 2]), setup))
##    fig = GLMakie.heatmap(obs)
##    fig |> display
##    for u in sol_ode.u
###        obs[] = u[Iu[i], i]
##        obs[] = INS.vorticity((u[:, :, 1], u[:, :, 2]), setup)
##        # fig |> display
##        sleep(0.05)
##    end
##end
##
### plots using Plots.jl
##using Plots, Printf
##anim = Animation()
##for (idx, (t, u)) in enumerate(zip(sol_node.t, sol_node.u))
##    ω = INS.vorticity((u[:, :, 1], u[:, :, 2]), setup)
##    title = @sprintf("Vorticity, t = %.3f", t)
##    fig = Plots.heatmap(ω'; xlabel = "x", ylabel = "y", title)
##    frame(anim, fig)
##end
##gif(anim, fps = 15)



########################
# Test the training
using Enzyme
using ComponentArrays
using SciMLSensitivity

pinit = ComponentArray(θ_node)
ode_data = Array(sol_ode)


# First test Enzyme for a one step a priori loss
#dual_P1 = (cache_F, cache_div, cache_p, psolver)
const U = stack(state.u) 
#dual_temp = copy(temp)
function fen(θ, u0, P1, temp)
    return sum(U .- dudt_out(temp, u0, (P1,θ), 0.0f0))
end
# Some tests of Enzyme
#dtheta = similar(pinit)
dtheta = Enzyme.make_zero(pinit)
#du =ones(Float32, size(u0stacked))
du = Enzyme.make_zero(u0stacked)
dual_P1 = Enzyme.make_zero(P1)
dual_temp = Enzyme.make_zero(temp)
# Compute the autodiff using Enzyme
Enzyme.autodiff(Enzyme.Reverse, fen, Active, Duplicated(pinit, dtheta), Duplicated(u0stacked, du), Duplicated(P1, dual_P1), Duplicated(temp, dual_temp))
# this shows us that Enzyme can differentiate our force. But what about SciML solvers?


# Then I can test Enzyme for a posteriori loss 
# pre-allcoate
pred = solve(prob_node, RK4(), u0 = u0stacked, p = (P1,pinit), saveat = saveat, sensealg=SciMLSensitivity.EnzymeVJP(), dt=dt)[1]
function myloss(θ::ComponentArray{Float32}, u0::Array{Float32}, P1)
    #pred = Array(solve(prob_node, RK4(), u0 = u0, p = (P1,θ), saveat = saveat, sensealg=SciMLSensitivity.EnzymeVJP(), dt=dt))
    pred = Array(solve(prob_node, RK4(), u0 = u0, p = (P1,θ), saveat = saveat, dt=dt))
    return sum(abs2, ode_data .- pred)
end
callback = function (θ, l; doplot = false)
    println(l)
    return false
end
callback(pinit, myloss(pinit, u0stacked, P1)...)
myloss(pinit, u0stacked, P1)

Enzyme.autodiff(Enzyme.Reverse, myloss, Active, Duplicated(pinit, dtheta), Duplicated(u0stacked, du), Duplicated(P1, dual_P1))
show(err)


#the function (loss) in enzyme needs to take the initial condition as an argument and it has to be dual in order to store the gradient

using SciMLSensitivity
using Optimization, OptimizationOptimisers, Optimisers
# Select the autodifferentiation type
#adtype = Optimization.AutoZygote()
adtype = Optimization.AutoEnzyme()
# We transform the NeuralODE into an optimization problem
optf = Optimization.OptimizationFunction((x, p) -> myloss(x), adtype);
optprob = Optimization.OptimizationProblem(optf, pinit);
# And train using Adam + clipping
ClipAdam = OptimiserChain(Adam(1.0f-1), ClipGrad(1));
algo = ClipAdam
algo = OptimizationOptimisers.Adam(1.0f-1)
#using OptimizationCMAEvolutionStrategy, Statistics
#algo = CMAEvolutionStrategyOpt();
#import OptimizationOptimJL: Optim
#algo = Optim.LBFGS();
# ** train loop
result_neuralode = Optimization.solve(
    optprob,
    algo,
    callback = callback,
    maxiters = 10)
# You can continue the training from here
pinit = result_neuralode.u
θ = pinit;
optprob = Optimization.OptimizationProblem(optf, pinit);


# You have to write the gradient such that it can be read by Enzyme, because it seems like there is no space to store the gradient or something 