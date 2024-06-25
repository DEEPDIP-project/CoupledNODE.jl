using ComponentArrays
using CUDA
using FFTW
using LinearAlgebra
using Optimisers
using Plots
using Printf
using Random
using DifferentialEquations
using JLD2
const ArrayType = Array
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
les_size = 64
dns_size = 128
les_size = 32
dns_size = 64
myseed = 1234
data_name = get_data_name(nu, les_size, dns_size, myseed)
# ***
# Do you want to plot the solution?
plotting = true

# ### Data generation
#
# Create some filtered DNS data (one initial condition only)
params_les = create_params(les_size; nu)
params_dns = create_params(dns_size; nu)
Random.seed!(myseed)

## Initial conditions
u0_dns = random_field(params_dns)

## Let's do some time stepping.
t = 0.0f0
dt_burn = 2.0f-4
dt = 2.0f-4
nt = 1000
tspan = (0.0f0, nburn * dt)
# with some burnout time to get rid of the initial condition
nburn = 1000
saveat_burn = 0.02f0
trange_burn = (0.0f0, nburn * dt_burn)

# run the burnout 
import DiffEqFlux: NeuralODE
include("./../../../src/NODE.jl")
F_dns(u) = Zygote.@ignore project(F_NS(u, params_dns), params_dns)
f_dns = create_f_CNODE((F_dns,); is_closed = false);
import Random, Lux
Random.seed!(123)
rng = Random.default_rng()
θ_dns, st_dns = Lux.setup(rng, f_dns);
dns = NeuralODE(f_dns,
    trange_burn,
    solver_algo,
    adaptive = false,
    dt = dt_burn,
    saveat = saveat_burn);
u_dns = dns(u0_dns, θ_dns, st_dns)[1]
u_dns = Array(dns(u0_dns, θ_dns, st_dns)[1]);

prob = ODEProblem(f, u, tspan, params_dns)
burn_sol = solve(prob, Tsit5(), adaptive = false, dt = dt, saveat = 0.02)

# and plot the burnout
if plotting
    plot()
    @gif for (idx, (t, u)) in enumerate(zip(u_dns.t, u_dns.u))
        ω = Array(vorticity(u, params_dns))
        title1 = @sprintf("Vorticity, burnout, t = %.3f", t)
        p1 = heatmap(ω'; xlabel = "x", ylabel = "y", titlefontsize = 11, title = title1)
        plot!(p1)
    end
end

#  fix from here

# Now we run the DNS and we compute the LES information at every time step
tspan = (0, nt * dt)
prob = ODEProblem(f, burn_sol.u[end], tspan, params_dns)
# to avoid memory crash, we delete the burnout solution
burn_sol = nothing
GC.gc()
# This is what we measure
v = zeros(Complex{Float32}, params_les.N, params_les.N, 2, nt + 1)
c = zeros(Complex{Float32}, params_les.N, params_les.N, 2, nt + 1)
# Now we can solve
sol = solve(prob, Tsit5(), adaptive = false, dt = dt, saveat = dt)

# then we loop to plot and compute LES
if plotting
    anim = Animation()
    for (idx, (t, u)) in enumerate(zip(sol.t, sol.u))
        ubar = spectral_cutoff(u, params_les.K)
        v[:, :, :, idx] = Array(ubar)
        c[:, :, :, idx] = Array(spectral_cutoff(F(u, params_dns), params_les.K) -
                                F(ubar, params_les))
        if idx % 10 == 0
            ω = Array(vorticity(u, params_dns))
            title = @sprintf("Vorticity, t = %.3f", t)
            fig = heatmap(ω'; xlabel = "x", ylabel = "y", title)
            frame(anim, fig)
        end
    end
    gif(anim)
else
    for (idx, (t, u)) in enumerate(zip(sol.t, sol.u))
        ubar = spectral_cutoff(u, params_les.K)
        v[:, :, :, idx] = Array(ubar)
        c[:, :, :, idx] = Array(spectral_cutoff(F(u, params_dns), params_les.K) -
                                F(ubar, params_les))
    end
end

# create the directory data if it does not exist
if !isdir("data")
    mkdir("data")
end

# Save all the simulation data
save("data/$(data_name).jld2", "data", Data(sol.t, sol.u, v, c, params_les, params_dns))
