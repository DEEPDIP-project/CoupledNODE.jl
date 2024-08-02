using CairoMakie
using IncompressibleNavierStokes
INS = IncompressibleNavierStokes


## make momentum! differentiable


# Setup and initial condition
T = Float32
ArrayType = Array
Re = T(1_000)
n = 128
n = 64
#n = 16
# this is the size of the domain, do not mix it with the time
lims = T(0), T(1)
x , y = LinRange(lims..., n + 1), LinRange(lims..., n + 1)
const setup = INS.Setup(x, y; Re, ArrayType);
ustart = INS.random_field(setup, T(0));
psolver = INS.psolver_direct(setup)
dt = T(1e-3)
trange = [T(0), T(1)]
savevery = 20
saveat = savevery * dt



# Solving using INS semi-implicit method
(state, outputs), time_ins, allocation, gc, memory_counters = @timed INS.solve_unsteady(;
    setup,
    ustart,
    tlims = trange,
    Δt = dt,
    #psolver = psolver,
    processors = (
        ehist = INS.realtimeplotter(;
            setup,
            plot = INS.energy_history_plot,
            nupdate = 10,
            displayfig = false
        ),
        field = INS.fieldsaver(; setup, nupdate = savevery),
        log = INS.timelogger(; nupdate = 100)
    )
);


############# Using SciML
using DifferentialEquations

# Projected force for SciML, to use in CNODE
F = similar(stack(ustart))
# and prepare a cache for the force
cache_F = (F[:,:,1], F[:,:,2])
cache_div = INS.divergence(ustart,setup)
cache_p = INS.pressure(ustart, nothing, 0.0f0, setup; psolver)
Ω = setup.grid.Ω


# Get the cache for the poisson solver
include("./test_redefinitions_INS.jl");
cache_ftemp, cache_ptemp, fact, cache_viewrange, cache_Ip = my_cache_psolver(setup.grid.x[1], setup)
# and use it to precompile an Enzyme-compatible psolver
my_psolve! = generate_psolver(cache_viewrange, cache_Ip, fact)

# In a similar way, get the function for the divergence 
mydivergence! = get_divergence!(cache_p, setup);
# and the function to apply the pressure
myapplypressure! = get_applypressure!(ustart, setup);


# Define the cache for the force 
using ComponentArrays
using KernelAbstractions
# I have also to take the grid size to stack into P
(; grid) = setup
(; Δ, Δu) = grid
P = ComponentArray(f=zeros(T, (n+2,n+2,2)),div=zeros(T,(n+2,n+2)), p=zeros(T,(n+2,n+2)), ft=zeros(T,size(cache_ftemp)), pt=zeros(T,size(cache_ptemp)), dz=zeros(T,(n+2,n+2)), Δ=stack(Δ))

const myzero = T(0)
# **********************8
# * Force in place
function F_ip(du, u, p, t)
    u_view = eachslice(u; dims = 3)
    F = eachslice(p.f; dims = 3)
    IncompressibleNavierStokes.apply_bc_u!(u_view, t, setup)
    IncompressibleNavierStokes.momentum!(F, u_view, nothing, t, setup)
    IncompressibleNavierStokes.apply_bc_u!(F, t, setup)
    mydivergence!(p.div, F, P.dz, P.Δ)
    @. p.div *= Ω
    my_psolve!(p.p, p.div, p.ft, p.pt)
    IncompressibleNavierStokes.apply_bc_p!(p.p, myzero, setup)
    myapplypressure!(F, p.p)
    IncompressibleNavierStokes.apply_bc_u!(F, t, setup)
    du[:,:,1] .= F[1]
    du[:,:,2] .= F[2]
    nothing
end;
temp = similar(stack(ustart));
F_ip(temp, stack(ustart), P, 0.0f0)
show(err)


# Solve the ODE using ODEProblem
prob = ODEProblem{true}(F_ip, stack(ustart), trange, p=P)
sol_ode, time_ode, allocation_ode, gc_ode, memory_counters_ode = @timed solve(
    prob,
    p = P,
    RK4();
    dt = dt,
    saveat = saveat,
);


# ------ Use Lux to create a dummy_NN
import Random, Lux;
Random.seed!(123);
rng = Random.default_rng();
dummy_NN = Lux.Chain(
    Lux.ReshapeLayer(((n+2)*(n+2),)),
    Lux.Dense((n+2)*(n+2)=>(n+2)*(n+2),init_weight = Lux.WeightInitializers.zeros32),
    Lux.ReshapeLayer(((n+2),(n+2))),
)
dummy_NN = Lux.Chain(
    Lux.ReshapeLayer(((n+2), (n+2), 1)),  # Add a channel dimension for the convolution
    Lux.Conv((3, 3), 1 => 1, pad=(1, 1), init_weight = Lux.WeightInitializers.ones32),  # 3x3 convolution with padding to maintain the input shape
    Lux.ReshapeLayer(((n+2), (n+2)))  # Remove the channel dimension
)
# Scale can not be differentiated by Enzyme!
#dummy_NN = Lux.Chain(
#    Lux.Scale((1,1)),
#)
θ_node, st_node = Lux.setup(rng, dummy_NN)

using ComponentArrays
θ_node = ComponentArray(θ_node)
# You can set it to 0 like this
#θ_node.weight = [0.0f0;;]
#θ_node.bias= [0.0f0;;]
Lux.apply(dummy_NN, stack(ustart), θ_node, st_node)[1]

P = ComponentArray(f=zeros(T, (n+2,n+2,2)),div=zeros(T,(n+2,n+2)), p=zeros(T,(n+2,n+2)), ft=zeros(T,size(cache_ftemp)), pt=zeros(T,size(cache_ptemp)), dz=zeros(T,(n+2,n+2)), Δ=stack(Δ), θ=copy(θ_node))

# Force+NN in-place version
dudt_nn(du, u, P, t) = begin
    F_ip(du, u, P, t) 
    du += Lux.apply(dummy_NN, u, P.θ , st_node)[1]
    nothing
end

dudt_nn(temp, stack(ustart), P, 0.0f0)
prob_node = ODEProblem{true}(dudt_nn, stack(ustart), trange, p=P)

u0stacked = stack(ustart)
sol_node, time_node, allocation_node, gc_node, memory_counters_node = @timed solve(prob_node, RK4(), u0 = u0stacked, p = P, saveat = saveat, dt=dt);


println("Done run")

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


# Compute the divergence of the final state
div_INS = INS.divergence(state.u, setup);
div_ode = INS.divergence((sol_ode.u[end][:,:,1],sol_ode.u[end][:,:,2]), setup);
div_node = INS.divergence((sol_node.u[end][:,:,1],sol_node.u[end][:,:,2]), setup);
p1 = Plots.heatmap(title="div_INS",div_INS)
p2 = Plots.heatmap(title="div_ODE",div_ode)
p3 = Plots.heatmap(title="div_NODE",div_node)
Plots.plot(p1, p2, p3, layout=(1,3), size=(900,300))



########################
# Test the autodiff using Enzyme 
using Enzyme
using ComponentArrays
using SciMLSensitivity


# First test Enzyme for something that does not make sense bu it has the structure of a priori loss
U = stack(state.u);
function fen(u0, p, temp, U)
    # Compute the force in-place
    #dudt_nn(temp, u0, p, 0.0f0)
    F_ip(temp, u0, p, 0.0f0)
    return sum(U - temp)
end
u0stacked = stack(ustart);
du = Enzyme.make_zero(u0stacked);
dP = Enzyme.make_zero(P);
temp = similar(stack(ustart));
dtemp = Enzyme.make_zero(temp);
dU = Enzyme.make_zero(U);
# Compute the autodiff using Enzyme
@timed Enzyme.autodiff(Enzyme.Reverse, fen, Active, DuplicatedNoNeed(u0stacked, du), DuplicatedNoNeed(P, dP), DuplicatedNoNeed(temp, dtemp), DuplicatedNoNeed(U, dU))
# the gradient that we need is only the following
dP.θ
# this shows us that Enzyme can differentiate our force. But what about SciML solvers?
println("Tested a priori")
show(err)



# Define a posteriori loss function that calls the ODE solver
# First, make a shorter run
# and remember to set a small dt
dt = T(1e-4)
typeof(dt)
trange = [T(0), T(3*dt)];
saveat = dt;
prob = ODEProblem{true}(F_ip, u0stacked, trange, p=P);
ode_data = Array(solve(prob, RK4(), u0 = u0stacked, p = P, saveat = saveat, dt=dt));
ode_data += T(0.1)*rand(Float32, size(ode_data))

# the loss has to be in place 
function loss(l,P, u0, pred, tspan, t, dt, target)
    myprob = ODEProblem{true}(dudt_nn, u0, tspan, p=P)
    pred .= Array(solve(myprob, RK4(), u0 = u0, p = P, saveat=t, dt=dt, verbose=false))
    l .= Float32(sum(abs2, target - pred))
    nothing
end
data = copy(ode_data);
target = copy(ode_data);
l=[T(0.0)];
loss(l,P, u0stacked, data, trange, saveat, dt, target);
l


# Test if the loss can be autodiffed
# [!] dl is called the 'seed' and it has to be marked to be one for correct gradient
l = [T(0.0)];
dl = Enzyme.make_zero(l) .+T(1);
dP = Enzyme.make_zero(P);
du = Enzyme.make_zero(u0stacked);
dd = Enzyme.make_zero(data);
dtarg = Enzyme.make_zero(target);
@timed Enzyme.autodiff(Enzyme.Reverse, loss, DuplicatedNoNeed(l, dl), DuplicatedNoNeed(P, dP), DuplicatedNoNeed(u0stacked, du), DuplicatedNoNeed(data, dd), Const(trange), Const(saveat), Const(dt), DuplicatedNoNeed(target, dtarg))
dP.θ
    


println("Now defining the gradient function")
extra_par = [u0stacked, data, dd, target, dtarg, trange, saveat, dt, du, dP, P];
Textra = typeof(extra_par);
Tth = typeof(P.θ);
function loss_gradient(G, extra_par) 
    u0, data, dd, target, dtarg, trange, saveat, dt, du0, dP, P = extra_par
    # [!] Notice that we are updating P.θ in-place in the loss function
    # Reset gradient to zero
    Enzyme.make_zero!(dP)
    # And remember to pass the seed to the loss funciton with the dual part set to 1
    Enzyme.autodiff(Enzyme.Reverse, loss, DuplicatedNoNeed([T(0)], [T(1)]), DuplicatedNoNeed(P,dP), DuplicatedNoNeed(u0, du0), DuplicatedNoNeed(data, dd) , Const(trange), Const(saveat), Const(dt), DuplicatedNoNeed(target, dtarg))
    # The gradient matters only for theta
    G .= dP.θ
    nothing
end

G = copy(dP.θ);
oo = loss_gradient(G, extra_par)


# This is to call loss using only P
#function over_loss(θ::Tth, p::TP)
function over_loss(θ, p)
    # Here we are updating P.θ in place
    p.θ .= θ
    loss(l,p, u0stacked, data, trange, saveat, dt, target);
    return l
end
callback = function (θ,l; doplot = false)
    println(l)
    return false
end
callback(P, over_loss(P.θ, P))


using SciMLSensitivity, Optimization, OptimizationOptimisers, Optimisers
optf = Optimization.OptimizationFunction((p,u)->over_loss(p,u[end]), grad=(G,p,e)->loss_gradient(G,e))
optprob = Optimization.OptimizationProblem(optf, P.θ, extra_par)


result_e, time_e, alloc_e, gc_e, mem_e = @timed Optimization.solve(optprob,
    OptimizationOptimisers.Adam(0.05),
    callback = callback,
    maxiters = 100)

