# # Comparison of solvers for 2D Navier-Stokes equations
# In this notebook we compare the *implementation and performance* of different solvers for the 2D Navier-Stokes equations. 
# We consider the following methods:
# * `IncompressibleNavierStokes.jl`
# * SciML, scpecifically `DifferentialEquations.jl`
# * `CoupledNODE.jl`: an in-house wrapper for SciML neural closures.

# Setup and initial condition
using GLMakie
using IncompressibleNavierStokes: IncompressibleNavierStokes as INS
T = Float64 # Precision is important, with Float32 SciML does not give a divergence-free solution.
ArrayType = Array
Re = T(1_000)
n = 64
lims = T(0), T(1)
x, y = LinRange(lims..., n + 1), LinRange(lims..., n + 1)
setup = INS.Setup(x, y; Re, ArrayType);
ustart = INS.random_field(setup, T(0));
psolver = INS.psolver_spectral(setup);
dt = T(1e-3)
trange = (T(0), T(10))
savevery = 20
saveat = savevery * dt;

# ## IncompressibleNavierStokes.jl solver
# We know that mathematically it is the best one.
# First, call to get the plots. 
# Look at the generated animation of the evolution of the vorticity field in time in plots/vorticity_INS.mkv.
(state, outputs), time_ins2, allocation2, gc2, memory_counters2 = @timed INS.solve_unsteady(;
    setup, ustart, tlims = trange, Δt = dt,
    processors = (
        ehist = INS.realtimeplotter(;
            setup, nupdate = 10, displayfig = false,
            plot = INS.energy_history_plot),
        anim = INS.animator(;
            setup, path = "./simulations/NavierStokes_2D/plots/vorticity_INS.mkv",
            nupdate = savevery),
        #espec = realtimeplotter(; setup, plot = energy_spectrum_plot, nupdate = 10),
        field = INS.fieldsaver(; setup, nupdate = savevery),
        log = INS.timelogger(; nupdate = 100)
    )
);

# Second, call to time it (without spending time in animations ~ 5% extra)
_, time_ins, allocation, gc, memory_counters = @timed INS.solve_unsteady(;
    setup, ustart, tlims = trange, Δt = dt);

# ## SciML via CoupledNODE.jl
# ### Projected force for SciML
# * Right-hand-side out-of-place
using CoupledNODE: NavierStokes
F_op = NavierStokes.create_right_hand_side(setup, psolver)

# * Right-hand-side in-place
F_ip = NavierStokes.create_right_hand_side_inplace(setup, psolver)

# Solve the ODE using `ODEProblem`. We use `RK4` (Runge-Kutta 4th order) because this same method is used in `IncompressibleNavierStokes.jl`.
using DifferentialEquations: ODEProblem, solve, RK4
prob_op = ODEProblem(F_op, stack(ustart), trange);
prob = ODEProblem{true}(F_ip, stack(ustart), trange);

# Solve the ODE using `ODEProblem`. We use `RK4` (Runge-Kutta 4th order) because this same method is used in `IncompressibleNavierStokes.jl`.
prob = ODEProblem{true}(F_ip, stack(ustart), trange);
sol_ode, time_ode, allocation_ode, gc_ode, memory_counters_ode = @timed solve(
    prob, RK4(); dt = dt, saveat = saveat, adaptive = false);

# Test the difference between in-place and out-of-place definitions
@time solve(prob_op, RK4(); dt = dt, saveat = saveat);
# Out of place: `64.082666 seconds (9.87 M allocations: 148.160 GiB, 4.25% gc time)`
# `65.306065 seconds (13.11 M allocations: 148.372 GiB, 3.76% gc time, 3.20% compilation time)`
@time solve(prob, RK4(); dt = dt, saveat = saveat);
# In place: `45.684604 seconds (8.14 M allocations: 503.659 MiB, 0.09% gc time)`
# `46.027974 seconds (8.20 M allocations: 507.999 MiB, 0.03% gc time, 0.08% compilation time)`

# Get timed solution to compare.
sol_ode, time_ode, allocation_ode, gc_ode, memory_counters_ode = @timed solve(
    prob, RK4(); dt = dt, saveat = saveat, adaptive = false);

# ## Comparison
# Bar plots comparing: time, memory allocation, and number of garbage collections (GC).
using Plots: Plots
p1 = Plots.bar(["INS", "CNODE"], [time_ins, time_ode],
    xlabel = "Method", ylabel = "Time (s)", title = "Time comparison", legend = false);
#Memory allocation
p2 = Plots.bar(["INS", "CNODE"],
    [memory_counters.allocd, memory_counters_ode.allocd],
    xlabel = "Method", ylabel = "Memory (bytes)",
    title = "Memory comparison", legend = false);
#Garbage collections
p3 = Plots.bar(["INS", "CNODE"], [gc, gc_ode], xlabel = "Method",
    ylabel = "Number of GC", title = "GC comparison", legend = false);

Plots.plot(p1, p2, p3, layout = (3, 1), size = (600, 800))

# ### Plots: final state of $u$
p1 = Plots.heatmap(title = "\$u\$ SciML CNODE", sol_ode.u[end][:, :, 1], ticks = false);
p2 = Plots.heatmap(title = "\$u\$ INS", state.u[1], ticks = false);
p3 = Plots.heatmap(title = "\$u_{INS}-u_{CNODE}\$",
    state.u[1] - sol_ode.u[end][:, :, 1], ticks = false);
Plots.plot(p1, p2, p3 = (1, 3), size = (900, 600), ticks = false)

# ### Plots: vorticity
vins = INS.vorticity((state.u[1], state.u[2]), setup)
vnode = INS.vorticity((sol_ode.u[end][:, :, 1], sol_ode.u[end][:, :, 2]), setup)
vor_lims = (-5, 5)
diff_lims = (-0.4, 0.4)

p1 = Plots.heatmap(title = "\$\\omega\$ in SciML CNODE", vnode,
    color = :viridis, ticks = false, clims = vor_lims);
p2 = Plots.heatmap(title = "vorticity \$(\\omega)\$ in INS", vins,
    color = :viridis, ticks = false, clims = vor_lims);
p3 = Plots.heatmap(title = "\$\\omega_{INS}-\\omega_{CNODE}\$",
    vins - vnode, clim = diff_lims, ticks = false);
Plots.plot(p1, p2, p3, layout = (1, 3), size = (900, 600))

# ### Divergence:

# INS
#div_INS = fill!(similar(setup.grid.x[1], setup.grid.N), 0)
#INS.divergence!(div_INS, state.u, setup)
div_INS = INS.divergence(state.u, setup)
maximum(abs.(div_INS))

# SciML via CNODE
u_last_ode = (sol_ode.u[end][:, :, 1], sol_ode.u[end][:, :, 2]);
div_ode = INS.divergence(u_last_ode, setup)
max_div_ode = maximum(abs.(div_ode))

using Printf: @sprintf
anim = Plots.Animation()
for (idx, (t, u)) in enumerate(zip(sol_ode.t, sol_ode.u))
    ∇_u = INS.divergence((u[:, :, 1], u[:, :, 2]), setup)
    title = @sprintf("\$\\nabla \\cdot u\$ SciML, t = %.3f s", t)
    fig = Plots.heatmap(∇_u'; xlabel = "x", ylabel = "y", title, aspect_ratio = :equal,
        ticks = false, size = (600, 600), clims = (-max_div_ode, max_div_ode))
    Plots.frame(anim, fig)
end
Plots.gif(anim, "simulations/NavierStokes_2D/plots/divergence_SciML.gif", fps = 15)

# **Conclusion:** While IncompressibleNavierStokes.jl guarantees a $\nabla \cdot u =0$ the other methods do not.

# ### Animations
# #### Animate SciML solution using `Makie`
# ! we can plot either the vorticity or the velocity field, however notice that the vorticity is 'flashing' (unstable)
let
    (; Iu) = setup.grid
    i = 1
    #obs = Observable(sol.u[1][Iu[i], i])
    obs = GLMakie.Observable(INS.vorticity(
        (sol_ode.u[1][:, :, 1], sol_ode.u[1][:, :, 2]), setup))
    fig = GLMakie.heatmap(obs, colorrange = vor_lims)
    fig |> display
    for u in sol_ode.u
        #obs[] = u[Iu[i], i]
        obs[] = INS.vorticity((u[:, :, 1], u[:, :, 2]), setup)
        #fig |> display
        sleep(0.05)
    end
end

# #### Animate SciML solution using `Plots.jl`
function animation_plots(; variable = "vorticity")
    anim = Plots.Animation()
    for (idx, (t, u)) in enumerate(zip(sol_ode.t, sol_ode.u))
        if variable == "vorticity"
            ω = INS.vorticity((u[:, :, 1], u[:, :, 2]), setup)
            title = @sprintf("Vorticity SciML, t = %.3f s", t)
            fig = Plots.heatmap(ω'; xlabel = "x", ylabel = "y", title, clims = vor_lims,
                color = :viridis, aspect_ratio = :equal, ticks = false, size = (600, 600))
        else
            title = @sprintf("\$u\$ SciML, t = %.3f s", t)
            fig = Plots.heatmap(u[:, :, 1]; xlabel = "x", ylabel = "y", title,
                aspect_ratio = :equal, ticks = false, size = (600, 600))
        end
        Plots.frame(anim, fig)
    end
    if variable == "vorticity"
        Plots.gif(anim, "simulations/NavierStokes_2D/plots/vorticity_SciML.gif", fps = 15)
    else
        Plots.gif(anim, "simulations/NavierStokes_2D/plots/velocity_SciML.gif", fps = 15)
    end
end

# Vorticity animation
animation_plots(; variable = "vorticity")

# Velocity animation
animation_plots(; variable = "velocity")

# #### Animate the difference in solution using `Plots.jl`
anim = Plots.Animation()
for idx in 1:Int(ceil(trange[end] / saveat))
    # the ODESolution saves the initial state too
    error_u = abs.(outputs.field[idx].u[1] - sol_ode.u[idx + 1][:, :, 1])
    title = @sprintf("\$|u_{INS}-u_{CNODE}|\$, t = %.3f s", sol_ode.t[idx])
    fig = Plots.heatmap(error_u; xlabel = "x", ylabel = "y", title,
        aspect_ratio = :equal, ticks = false, size = (600, 600))
    Plots.frame(anim, fig)
end
Plots.gif(anim, "simulations/NavierStokes_2D/plots/u_INS-SciML.gif", fps = 15)

# ## Testing
dt_1 = T(1e-3)
trange_1 = (T(0), dt_1)
state_1, outputs_1 = INS.solve_unsteady(; setup, ustart, tlims = trange_1, Δt = dt_1)

using DifferentialEquations: Tsit5
prob_1 = ODEProblem{true}(F_ip, stack(ustart), trange_1);
sol_ode_1 = solve(
    prob_1, Tsit5(); dt = dt_1, saveat = dt_1, adaptive = false);

sol_ode_1.u[1] == stack(ustart) #first element is the initial condition
size(sol_ode_1.u[end][:, :, 1])
size(state_1.u[1])

# the difference between the solutions is not zero but is very little:
sol_ode_1.u[end][:, :, 1] == state_1.u[1]
maximum(abs.(sol_ode_1.u[end][:, :, 1] - state_1.u[1]))

# Let's check the divergence
#INS
div_INS_1 = INS.divergence(state_1.u, setup)
maximum(abs.(div_INS_1))
#this is supprigingly larger than expected with dt=1e-3, but is again almost zero if the time step is reduced to 1e-4

#SciML
u_last_ode_1 = (sol_ode_1.u[end][:, :, 1], sol_ode_1.u[end][:, :, 2]);
div_ode_1 = INS.divergence(u_last_ode_1, setup)
max_div_ode_1 = maximum(abs.(div_ode_1))
