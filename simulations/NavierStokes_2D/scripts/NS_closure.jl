# Goal: Gather all the elements that we create a closure of the Navier-Stokes equations using INS and SciML.
# In this case we are not going to use any elements from Neural Closure.
using CairoMakie
import IncompressibleNavierStokes as INS
import Random
Random.seed!(123);

# Define the problem
T = Float32
ArrayType = Array
Re = T(1_000)
lims = T(0), T(1);
dt = T(1e-3);
trange = [T(0), T(1)]
saveat = [i * dt for i in 1:div(1, dt)]

# dns and les grid 
n_dns = 128
N_dns = n_dns + 2
n_les = 64
N_les = n_les + 2
x_dns, y_dns = LinRange(lims..., n_dns + 1), LinRange(lims..., n_dns + 1);
x_les, y_les = LinRange(lims..., n_les + 1), LinRange(lims..., n_les + 1);
setup_dns = INS.Setup(x_dns, y_dns; Re, ArrayType);
setup_les = INS.Setup(x_les, y_les; Re, ArrayType);

# Filter
import SparseArrays
function create_filter_matrix_2d(
        dx_dns, dx_les, N_dns, N_les, ΔΦ, kernel_type, MY_TYPE = Float64)
    #Filter kernels
    gaussian(Δ, x, y) = MY_TYPE(sqrt(6 / π) / Δ * exp(-6 * (x^2 + y^2) / Δ^2))
    top_hat(Δ, x, y) = MY_TYPE((abs(x) ≤ Δ / 2) * (abs(y) ≤ Δ / 2) / (Δ^2))

    #Choose kernel
    kernel = kernel_type == "gaussian" ? gaussian : top_hat

    x_dns = collect(0:dx_dns:((N_dns - 1) * dx_dns))
    y_dns = collect(0:dx_dns:((N_dns - 1) * dx_dns))
    x_les = collect(0:dx_les:((N_les - 1) * dx_les))
    y_les = collect(0:dx_les:((N_les - 1) * dx_les))

    #Discrete filter matrix (with periodic extension and threshold for sparsity)
    Φ = sum(-1:1) do z_x
        sum(-1:1) do z_y
            d_x = @. x_les - x_dns' - z_x
            d_y = @. y_les - y_dns' - z_y
            if kernel_type == "gaussian"
                @. kernel(ΔΦ, d_x, d_y) * (abs(d_x) ≤ 3 / 2 * ΔΦ) * (abs(d_y) ≤ 3 / 2 * ΔΦ)
            else
                @. kernel(ΔΦ, d_x, d_y)
            end
        end
    end
    Φ = Φ ./ sum(Φ; dims = 2) #Normalize weights
    Φ = SparseArrays.sparse(Φ)
    SparseArrays.dropzeros!(Φ)
    return Φ
end

dx_dns = x_dns[2] - x_dns[1]
dx_les = x_les[2] - x_les[1]
# To get the LES, we use a Gaussian filter kernel, truncated to zero outside of $3 / 2$ filter widths.
ΔΦ = 5 * dx_les
Φ = create_filter_matrix_2d(dx_dns, dx_les, N_dns, N_les, ΔΦ, "gaussian", T)
function apply_filter(ϕ, u)
    vx = ϕ * u[1] * ϕ'
    vy = ϕ * u[2] * ϕ'
    return (vx, vy)
end

# Initial condition
u0_dns = INS.random_field(setup_dns, T(0));
@assert u0_dns[1][1, :] == u0_dns[1][end - 1, :]
@assert u0_dns[1][end, :] == u0_dns[1][2, :]
#u0_les = apply_filter(Φ, (u0_dns[1][2:end-1, 2:end-1], u0_dns[2][2:end-1, 2:end-1])) 
u0_les = apply_filter(Φ, u0_dns)
@assert u0_les[1][1, :] == u0_les[1][end - 1, :]
@assert u0_les[1][end, :] == u0_les[1][2, :]

import Plots
p1 = Plots.heatmap(u0_dns[1], title = "DNS initial condition")
p2 = Plots.heatmap(u0_les[1], title = "LES initial condition")
Plots.plot(p1, p2, layout = (1, 2), size = (800, 400))

# Get the forces for NS
# Zygote force (out-of-place)
import CoupledNODE: create_right_hand_side
F_out_dns = create_right_hand_side(setup_dns, INS.psolver_direct(setup_dns));
F_out_les = create_right_hand_side(setup_les, INS.psolver_direct(setup_les));

# define sciml problems
import DifferentialEquations: solve, ODEProblem, Tsit5
prob_dns = ODEProblem(F_out_dns, stack(u0_dns), trange)
prob_les = ODEProblem(F_out_les, stack(u0_les), trange)

# Solve the exact solutions
sol_dns, time_dns, allocation_dns, gc_dns, memory_counters_dns = @timed solve(
    prob_dns, Tsit5(); dt = dt, saveat = saveat);
sol_les, time_les, allocation_les, gc_les, memory_counters_les = @timed solve(
    prob_les, Tsit5(); dt = dt, saveat = saveat);

# Compare the times of the different methods via a bar plot
p1 = Plots.bar(["DNS", "LES"], [time_dns, time_les], xlabel = "Method",
    ylabel = "Time (s)", title = "Time comparison")
# Compare the memory allocation
p2 = Plots.bar(["DNS", "LES"], [memory_counters_dns.allocd, memory_counters_les.allocd],
    xlabel = "Method", ylabel = "Memory (bytes)", title = "Memory comparison")
# Compare the number of garbage collections
p3 = Plots.bar(["DNS", "LES"], [gc_dns, gc_les], xlabel = "Method",
    ylabel = "Number of GC", title = "GC comparison")
Plots.plot(p1, p2, p3, layout = (3, 1), size = (600, 800), legend = false)

# and show an animation of the solution
anim = Plots.Animation()
fig = Plots.plot(layout = (3, 1), size = (300, 800))
Plots.@gif for i in 1:10:length(saveat)
    p1 = Plots.heatmap(sol_dns.u[i][:, :, 1], title = "DNS")
    p2 = Plots.heatmap(sol_les.u[i][:, :, 1], title = "LES")
    uu = (sol_dns.u[i][:, :, 1], sol_dns.u[i][:, :, 2])
    u_filtered = apply_filter(Φ, uu)
    p3 = Plots.heatmap(u_filtered[1], title = "Filtered DNS")
    title = "Time: $(round((i - 1) * dt, digits = 2))"
    fig = Plots.plot(p1, p2, p3, size = (300, 800), layout = (3, 1), suptitle = title)
    Plots.frame(anim, fig)
end

# ## A-priori fitting
# Target: filtered dns force. Can be used to evaulate the overall performance of the NN
all_F_filtered = []
for i in 1:length(saveat)
    F_dns = F_out_dns(sol_dns.u[i], nothing, 0)
    F_filt = apply_filter(Φ, (F_dns[:, :, 1], F_dns[:, :, 2]))
    push!(all_F_filtered, stack(F_filt))
end

# define a dummy_NN to train
import Lux
dummy_NN = Lux.Chain(
    x -> view(x, :, :, :, :),
    Lux.Conv((3, 3), 2 => 2; pad = (1, 1), stride = (1, 1)),
    Lux.Conv((3, 3), 2 => 2; pad = (1, 1), stride = (1, 1)),
    Lux.Conv((3, 3), 2 => 2; pad = (1, 1), stride = (1, 1))
)

import ComponentArrays: ComponentArray
rng = Random.default_rng();
θ, st = Lux.setup(rng, dummy_NN)
θ = ComponentArray(θ)

# Define the right hand side function with the neural network closure   
dudt_nn(u, θ, t) = begin
    F_out_les(u, θ, t) .+ Lux.apply(dummy_NN, u, θ, st)[1][:, :, :, 1]
end
dudt_nn(stack(u0_les), θ, T(0))

# Define the loss function
import Zygote
npoints = 32
function loss_priori(p)
    l, d = 0, 0
    #Select a random set of points 
    for i in 1:npoints
        i = Zygote.@ignore rand(1:length(saveat))
        l += sum(abs2, all_F_filtered[i] .- dudt_nn(sol_les.u[i], p, 0.0))
        d += sum(abs2, all_F_filtered[i])
    end
    return l / d
end

# Train using Zygote
import CoupledNODE: callback
import Optimization, OptimizationOptimisers
optf = Optimization.OptimizationFunction(
    (x, p) -> loss_priori(x), Optimization.AutoZygote())
optprob = Optimization.OptimizationProblem(optf, θ)

result_priori, time_priori, alloc_priori, gc_priori, mem_priori = @timed Optimization.solve(
    optprob,
    OptimizationOptimisers.Adam(0.1);
    callback = callback,
    maxiters = 50,
    progress = true
)
θ_priori = result_priori.u

# ## A-posteriori fitting
# get the filtered solution
u_filtered = []
for i in 1:length(sol_dns.u)
    u = (sol_dns.u[i][:, :, 1], sol_dns.u[i][:, :, 2])
    push!(u_filtered, stack(apply_filter(Φ, u)))
end

#re-initialization of the dummy network
θ, st = Lux.setup(rng, dummy_NN)
θ = ComponentArray(θ)

# Define the loss (a-posteriori)
nunroll = 5
saveat_loss = [i * dt for i in 1:nunroll]
tspan = [T(0), T(nunroll * dt)]
function loss_posteriori_Z(p)
    i0 = Zygote.@ignore rand(1:(length(saveat) - nunroll))
    prob = ODEProblem(dudt_nn, u_filtered[i0], tspan, p)
    pred = Array(solve(prob, Tsit5(); u0 = u_filtered[i0], p = p, saveat = saveat_loss))
    # remember to discard sol at i0
    return T(sum(abs2, stack(u_filtered[(i0 + 1):(i0 + nunroll)]) - pred) /
             sum(abs2, stack(u_filtered[(i0 + 1):(i0 + nunroll)])))
end

# train a-posteriori
optf = Optimization.OptimizationFunction(
    (x, p) -> loss_posteriori_Z(x), Optimization.AutoZygote()
)
optprob = Optimization.OptimizationProblem(optf, θ)

using SciMLSensitivity #for compatibility with reverse mode
result_posteriori, time_posteriori, alloc_posteriori, gc_posteriori, mem_posteriori = @timed Optimization.solve(
    optprob,
    OptimizationOptimisers.Adam(0.1),
    callback = callback,
    maxiters = 50,
    progress = true
)
θ_posteriori_Z = result_posteriori.u
