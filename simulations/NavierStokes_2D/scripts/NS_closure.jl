using CairoMakie
using IncompressibleNavierStokes
INS = IncompressibleNavierStokes

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
trange = (T(0), T(10))
savevery = 20
saveat = 20 * dt

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
# TODO: can we store history of u for comparisons? probably not


############# Using SciML
using DifferentialEquations

# Projected force for SciML, to use in CNODE
F = similar(stack(ustart))
cache_F = (F[:,:,1], F[:,:,2])
cache_div = INS.divergence(ustart,setup)
cache_p = INS.pressure(ustart, nothing, 0.0f0, setup; psolver)
cache_out = similar(F)
create_right_hand_side(setup, psolver, cache_F, cache_div, cache_p, cache_out ) = function right_hand_side(u, p=nothing, t=nothing)
    u = eachslice(u; dims = 3)
    INS.apply_bc_u!(u, t, setup)
    INS.momentum!(cache_F, u, nothing, t, setup)
    INS.apply_bc_u!(cache_F, t, setup; dudt = true)
    INS.project!(cache_F, setup; psolver, div=cache_div, p=cache_p)
    INS.apply_bc_u!(cache_F, t, setup; dudt = true)
    return stack(cache_F)
end
# test the forces
# Note: Requires `stack(u)` to create one array
F_syv = create_right_hand_side(setup, psolver, cache_F, cache_div, cache_p, cache_out);
F_syv(stack(ustart), nothing, 0.0f0);
# * There is also an in-place version
create_rhs_ip(setup, psolver, cache_F, cache_div, cache_p, cache_out ) = function right_hand_side(du, u, p=nothing, t=nothing)
    u = eachslice(u; dims = 3)
    INS.apply_bc_u!(u, t, setup)
    INS.momentum!(cache_F, u, nothing, t, setup)
    INS.apply_bc_u!(cache_F, t, setup; dudt = true)
    INS.project!(cache_F, setup; psolver, div=cache_div, p=cache_p)
    INS.apply_bc_u!(cache_F, t, setup; dudt = true)
    du[:,:,1] = cache_F[1]
    du[:,:,2] = cache_F[2]
    nothing
end
temp = similar(stack(ustart))
F_ip = create_rhs_ip(setup, psolver, cache_F, cache_div, cache_p, cache_out);
F_ip(temp, stack(ustart), nothing, 0.0f0)



# Solve the ODE using ODEProblem
prob = ODEProblem(F_syv, stack(ustart), trange)
prob = ODEProblem{true}(F_ip, stack(ustart), trange)
sol_ode, time_ode, allocation_ode, gc_ode, memory_counters_ode = @timed solve(
    prob,
    RK4();
    dt = dt,
    saveat = saveat,
);
# Or solve the ODE using NeuralODE and CNODE
# [!] is NODE compatible with the in-place version?
import DiffEqFlux: NeuralODE;
include("./../../../src/NODE.jl");
f_dns = create_f_CNODE((F_syv,); is_closed = false);
import Random, Lux;
Random.seed!(123);
rng = Random.default_rng();
θ_dns, st_dns = Lux.setup(rng, f_dns);
# Now we run the DNS and we compute the LES information at every time step
dns = NeuralODE(f_dns,
    trange,
    RK4(),
    adaptive = false,
    dt = dt,
    saveat = saveat);
sol_node, time_node, allocation_node, gc_node, memory_counters_node = @timed dns(stack(ustart), θ_dns, st_dns)[1];
# Alternatively, we can follow this example to get a node using ODEProblem and in-place updates
# https://docs.sciml.ai/SciMLSensitivity/stable/examples/neural_ode/neural_ode_flux/
import Random, Lux;
Random.seed!(123);
rng = Random.default_rng();
dummy_NN = Chain(
    u -> let ux=u[1], uy=u[2]
        (ux .*0, uy .*0)
    end
)
NN_closure = Parallel(nothing, dummy_NN)
Fnode = Chain(
            SkipConnection(
            NN_closure,
            (f_NN, uv) -> begin
                println(size(F_syv(uv))
                println(size(stack(f_NN))
                F_syv(uv) + stack(f_NN)
                end
            ; name = "Closure"),
        )
θ_node, st_node = Lux.setup(rng, Fnode);
dudt(u, θ, t) = Lux.apply(Fnode, u, θ, st_node) # need to restrcture for backprop!
prob_node = ODEProblem(dudt, stack(ustart), trange)

#sol_node, time_node, allocation_node, gc_node, memory_counters_node = @timed solve(prob_node, RK4(), u0 = stack(ustart), p = θ_node, saveat = saveat, dt=dt);
sol_aa, time_aa, allocation_aa, gc_aa, memory_counters_aa = @timed solve(prob_node, RK4(), u0 = stack(ustart), p = θ_node, saveat = saveat, dt=dt);


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



# Animate solution using Makie
# ! we can plot either the vorticity or the velocity field, however notice that the vorticity is 'flashing' (unstable)
using GLMakie
let
    (; Iu) = setup.grid
    i = 1
#    obs = Observable(sol.u[1][Iu[i], i])
#    ω = IncompressibleNavierStokes.vorticity((u[:, :, 1], u[:, :, 2]), setup)
    obs = Observable(INS.vorticity((sol_ode.u[1][:, :, 1], sol_ode.u[1][:, :, 2]), setup))
    fig = GLMakie.heatmap(obs)
    fig |> display
    for u in sol_ode.u
#        obs[] = u[Iu[i], i]
        obs[] = INS.vorticity((u[:, :, 1], u[:, :, 2]), setup)
        # fig |> display
        sleep(0.05)
    end
end

# plots using Plots.jl
using Plots, Printf
anim = Animation()
for (idx, (t, u)) in enumerate(zip(sol_ode.t, sol_ode.u))
    ω = INS.vorticity((u[:, :, 1], u[:, :, 2]), setup)
    title = @sprintf("Vorticity, t = %.3f", t)
    fig = Plots.heatmap(ω'; xlabel = "x", ylabel = "y", title)
    frame(anim, fig)
end
gif(anim, fps = 15)

