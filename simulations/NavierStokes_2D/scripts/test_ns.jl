using CairoMakie
using IncompressibleNavierStokes
output = "output/test_F"
T = Float32
ArrayType = Array
Re = T(1_000)
n = 256
lims = T(0), T(1)
x = LinRange(lims..., n + 1), LinRange(lims..., n + 1)
setup = Setup(x...; Re, ArrayType);
ustart = random_field(setup, T(0));
dt = T(1e-3)
trange = (T(0), T(1e-3))
trange = (T(0), T(5))
savevery = 20
saveat = 20 * dt

@time state, outputs = solve_unsteady(;
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
        espec = realtimeplotter(; setup, plot = energy_spectrum_plot, nupdate = 10),
        log = timelogger(; nupdate = 100)
    )
);
outputs.anim

# Time to reach t=1:
# 52.651788 seconds (52.93 M allocations: 4.053 GiB, 1.59% gc time, 68.82% compilation time: <1% of which was recompilation)
# 26.602432 seconds (53.02 M allocations: 4.059 GiB, 2.45% gc time, 85.26% compilation time: <1% of which was recompilation)
# 27.747093 seconds (53.03 M allocations: 4.060 GiB, 2.50% gc time, 83.14% compilation time: <1% of which was recompilation)
# Time to reach t=5:
# 72.089871 seconds (120.69 M allocations: 10.436 GiB, 1.53% gc time, 31.12% compilation time: <1% of which was recompilation)

# This is the force ! [https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/8ae287a20b67b3c808dd36cf257903efb6eedb30/src/operators.jl#L1005]
# (there is also an in-place version)
F = IncompressibleNavierStokes.momentum(state.u, nothing, state.t, setup)
# To this I have to add 
psolver = IncompressibleNavierStokes.default_psolver(setup)
p = IncompressibleNavierStokes.pressure(state.u, nothing, state.t, setup; psolver = psolver)
Gp = IncompressibleNavierStokes.pressuregradient(p, setup)
div = IncompressibleNavierStokes.divergence(F, setup)

using Zygote

out = similar(u0_dns)
f_c = similar(u0_dns)
f_c = zeros(size(u0_dns))
f_c = @SVector zeros(size(u0_dns))
tuple(f_c[:, :, 1:2])
using StaticArrays
dF = MArray{2, Float32}(f_c)
uu = eachslice(u0_dns, dims = 3)
dF .= IncompressibleNavierStokes.momentum(uu, nothing, 0.0f0, setup)
# ##########
# [!] YOU Can NOT Zygote.ignore the force, otherwise you can not do a posteriori fitting!
# ##########
F_syv(u) = begin
    u = eachslice(u, dims = 3)
    IncompressibleNavierStokes.apply_bc_u!(u, 0.0f0, setup)
    F = IncompressibleNavierStokes.momentum(u, nothing, 0.0f0, setup)
    IncompressibleNavierStokes.apply_bc_u!(F, 0.0f0, setup)
    F = IncompressibleNavierStokes.project(F, setup, psolver = psolver)
    IncompressibleNavierStokes.apply_bc_u!(F, 0.0f0, setup)
    out[:, :, 1] .= F[1]
    out[:, :, 2] .= F[2]
    out
end
F_syv2(u) = begin
    u = eachslice(u, dims = 3)
    IncompressibleNavierStokes.apply_bc_u!(u, 0.0f0, setup)
    F = IncompressibleNavierStokes.momentum(u, nothing, 0.0f0, setup)
    IncompressibleNavierStokes.apply_bc_u!(F, 0.0f0, setup)
    div = IncompressibleNavierStokes.divergence(F, setup)
    div = @. div * setup.grid.Ω
    p = IncompressibleNavierStokes.poisson(psolver, div)
    IncompressibleNavierStokes.apply_bc_p!(p, 0.0f0, setup)
    Gp = IncompressibleNavierStokes.pressuregradient(p, setup)
    rhs = @. F .- Gp
    IncompressibleNavierStokes.apply_bc_u!(rhs, 0.0f0, setup)
    out[:, :, 1] .= rhs[1]
    out[:, :, 2] .= rhs[2]
    out
end

using Statistics
a = F_syv(u0_dns);
b = F_syv2(u0_dns);
a == b
########################
########################
########################
########################

using ComponentArrays
using CUDA
using FFTW
using LinearAlgebra
using Optimisers
using Plots
using Printf
using Random
using DifferentialEquations
using Zygote
using JLD2
using PreallocationTools
#const ArrayType = Array
const solver_algo = Tsit5()
const MY_TYPE = Float32 # use float32 if you plan to use a GPU
import CUDA # Test if CUDA is running
if CUDA.functional()
    CUDA.allowscalar(false)
    const ArrayType = CuArray
    import DiffEqGPU: GPUTsit5
    const solver_algo = GPUTsit5()
end
z = CUDA.functional() ? CUDA.zeros : (s...) -> zeros(MY_TYPE, s...)

include("./../../../src/NavierStokes.jl")

# Lux likes to toss random number generators around, for reproducible science
rng = Random.default_rng()

# This line makes sure that we don't do accidental CPU stuff while things
# should be on the GPU
CUDA.allowscalar(false)

## *** Generate the data with the following parameters
nu = 5.0f-4
nu = 1.0f0 / Re

myseed = 2406
plotting = true

# ### Data generation
#
# Create some filtered DNS data (one initial condition only)
params_dns = create_params(129; nu)
Random.seed!(myseed)

## Initial conditions
nsamp = 1
u0_dns = random_spectral_field(params_dns, nsamp = nsamp)
# and we make sure that the initial condition is divergence free
maximum(abs, params_dns.k .* u0_dns[:, :, 1, :] .+ params_dns.k' .* u0_dns[:, :, 2, :])

u0_dns = ustart
u0_dns = permutedims(cat(u0_dns[1], u0_dns[2], dims = 4), (1, 2, 4, 3))

# Create that cache which is used to compute the right hand side of the Navier-Stokes
cache = create_cache(u0_dns);

## Let's do some time stepping.
import DiffEqFlux: NeuralODE
include("./../../../src/NODE.jl")
#F_dns(u) = Zygote.@ignore project(F_NS(u, params_dns, cache), params_dns, cache)
f_dns = create_f_CNODE((F_syv,); is_closed = false);
import Random, Lux
Random.seed!(123)
rng = Random.default_rng()
θ_dns, st_dns = Lux.setup(rng, f_dns);
# Now we run the DNS and we compute the LES information at every time step
dns = NeuralODE(f_dns,
    trange,
    solver_algo,
    adaptive = false,
    dt = dt,
    saveat = saveat);
@time sol = dns(u0_dns, θ_dns, st_dns)[1]
# Time to reach t=1:
# 31.814979 seconds (35.79 M allocations: 58.981 GiB, 2.72% gc time, 15.56% compilation time: 4% of which was recompilation)
# 24.996927 seconds (35.79 M allocations: 58.981 GiB, 2.99% gc time, 19.44% compilation time: 4% of which was recompilation)
# 38.798794 seconds (35.86 M allocations: 58.985 GiB, 2.17% gc time, 13.47% compilation time: 4% of which was recompilation)
# Time to reach t=5:
# 178.680643 seconds (162.00 M allocations: 241.369 GiB, 2.06% gc time, 0.38% compilation time: 76% of which was recompilation)
# 172.312957 seconds (162.55 M allocations: 248.847 GiB, 1.38% gc time, 0.32% compilation time)
# TOOO SLOW !

sol.u[end]
state.u[1]

z = IncompressibleNavierStokes.apply_bc_u!(state.u, state.t, setup)
z2 = IncompressibleNavierStokes.apply_bc_u!(
    (sol.u[end][:, :, 1], sol.u[end][:, :, 2]), state.t, setup)

Plots.heatmap(sol.u[end][:, :, 1])
Plots.heatmap(state.u[1])
Plots.heatmap(z[1])
Plots.heatmap(z2[1])

Plots.heatmap(state.u[1] - sol.u[end][:, :, 1])
Plots.heatmap(state.u[1] - z[1])
Plots.heatmap(state.u[1] - z2[1])
Plots.heatmap(z[1] - sol.u[end][:, :, 1])
Plots.heatmap(z2[1] - sol.u[end][:, :, 1])
Plots.heatmap(z2[1] - z[1])

using Plots
# then we loop to plot and compute LES
if plotting
    anim = Animation()
end
for (idx, (t, u)) in enumerate(zip(sol.t, sol.u))
    if plotting #&& (idx % 10 == 0)
        ω = IncompressibleNavierStokes.vorticity((u[:, :, 1], u[:, :, 2]), setup)
        title = @sprintf("Vorticity, t = %.3f", t)
        fig = Plots.heatmap(ω'; xlabel = "x", ylabel = "y", title)
        frame(anim, fig)
    end
end
if plotting
    gif(anim, fps = 15)
end

# Alternative approach:
# this is doing an Euler step and then a Poisson step and then does the corrected force
# (this is not guaranteed to make sense)

function f2(du, u, p, t)
    ut = (u[:, :, 1], u[:, :, 2]) #make as view
    ut = IncompressibleNavierStokes.apply_bc_u!(ut, 0.0f0, setup)
    # Do a fake time step
    m = IncompressibleNavierStokes.momentum(ut, nothing, 0.0f0, setup)
    m = IncompressibleNavierStokes.apply_bc_u!(m, 0.0f0, setup)
    up = @. ut .+ dt .* m
    # Solve Poisson equation after the fake time step
    up = IncompressibleNavierStokes.apply_bc_u!(up, 0.0f0, setup)
    F = IncompressibleNavierStokes.momentum(up, nothing, 0.0f0, setup)
    F = IncompressibleNavierStokes.apply_bc_u!(F, 0.0f0, setup)
    div = IncompressibleNavierStokes.divergence(F, setup)
    div = @. div * setup.grid.Ω
    pr = IncompressibleNavierStokes.poisson(psolver, div)
    pr = IncompressibleNavierStokes.apply_bc_p(pr, 0.0f0, setup)
    Gp = IncompressibleNavierStokes.pressuregradient(pr, setup)
    # Correct the force with the pressure gradient
    rhs = @. m .- Gp
    rhs = IncompressibleNavierStokes.apply_bc_u!(rhs, 0.0f0, setup)
    du[:, :, 1] .= rhs[1]
    du[:, :, 2] .= rhs[2]
end

a = similar(u0_dns)
f2(a, u0_dns, nothing, 0.0f0)
q = F_syv(u0_dns)
Plots.heatmap(a[:, :, 1] - q[:, :, 1])
Plots.heatmap(a[:, :, 2] - q[:, :, 2])

prob2 = ODEProblem(f2, u0_dns, trange, nothing)
# We solve using Runge-Kutta 4 to compare with the direct implementation
@time sol2 = solve(prob2, Tsit5(), adaptive = false, dt = dt, saveat = saveat)
# Time to reach t=1:
# 34.558213 seconds (40.45 M allocations: 47.648 GiB, 1.79% gc time, 2.23% compilation time)

if plotting
    anim = Animation()
end
for (idx, (t, u)) in enumerate(zip(sol2.t, sol2.u))
    if plotting #&& (idx % 10 == 0)
        ω = IncompressibleNavierStokes.vorticity((u[:, :, 1], u[:, :, 2]), setup)
        title = @sprintf("Vorticity, t = %.3f", t)
        #fig = Plots.heatmap(ω'; xlabel = "x", ylabel = "y", title)
        fig = Plots.heatmap(u[:, :, 1]; xlabel = "x", ylabel = "y", title)
        frame(anim, fig)
    end
end
if plotting
    gif(anim, fps = 15)
end

# [!] the vorticity is flashing, check why

# This formulation attempts to define the problem as a DAE and let SciML handle the time stepping
# I do not trust this
function fdae(out, du, u, p, t)
    u = (u[:, :, 1], u[:, :, 2]) #make as view
    u = IncompressibleNavierStokes.apply_bc_u!(u, 0.0f0, setup)
    F = IncompressibleNavierStokes.momentum(u, nothing, 0.0f0, setup)
    F = IncompressibleNavierStokes.apply_bc_u!(F, 0.0f0, setup)
    div = IncompressibleNavierStokes.divergence(F, setup)
    div = @. div * setup.grid.Ω
    press = IncompressibleNavierStokes.poisson(psolver, div)
    press = IncompressibleNavierStokes.apply_bc_p(press, 0.0f0, setup)
    Gp = IncompressibleNavierStokes.pressuregradient(p, setup)
    rhs = @. F .- Gp
    rhs = IncompressibleNavierStokes.apply_bc_u!(rhs, 0.0f0, setup)
    #permutedims(cat(rhs[1], rhs[2], dims=4), (1,2,4,3))

    divu = IncompressibleNavierStokes.divergence(u, setup)
    divu = @. divu * setup.grid.Ω

    out[1] = rhs[1] - du[1]
    out[2] = rhs[2] - du[2]
    out[3] = divu
end

differential_vars = [true, true, false]
u0_dns
u0 = [u0_dns[:, :, 1], u0_dns[:, :, 2], 0]
q = F_syv(u0_dns)
du0 = [q[:, :, 1], q[:, :, 2], 0]
prob = DAEProblem(fdae, du0, u0, trange, differential_vars = differential_vars)

sol3 = solve(prob, Tsit5(), adaptive = false, dt = dt, saveat = saveat)
