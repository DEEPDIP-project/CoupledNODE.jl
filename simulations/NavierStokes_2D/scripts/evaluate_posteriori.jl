using CoupledNODE: cnn
using CoupledNODE.NavierStokes: create_right_hand_side_with_closure
using DifferentialEquations: ODEProblem, solve, Tsit5
using IncompressibleNavierStokes: IncompressibleNavierStokes as INS
using JLD2: load, @load
using Plots: Plots
using Printf: @sprintf
using Random: Random

T = Float32
ArrayType = Array
rng = Random.Xoshiro(123)
ig = 1 # index of the LES grid to use.

# Load model params
outdir = "simulations/NavierStokes_2D/outputs"
@load "$outdir/trained_model_posteriori.jld2" θ_posteriori st

# Load data params
params = load("simulations/NavierStokes_2D/data/params_data.jld2", "params")
D = params.D # dimension
# Build LES setups and assemble operators
setups = map(params.nles) do nles
    x = ntuple(α -> LinRange(T(0.0), T(1.0), nles + 1), params.D)
    INS.Setup(; x = x, Re = params.Re)
end

# Create model
closure, _,
_ = cnn(;
    T = T,
    D = D,
    data_ch = D,
    radii = [3, 3],
    channels = [2, 2],
    activations = [tanh, identity],
    use_bias = [false, false],
    rng
)

dudt_nn = create_right_hand_side_with_closure(
    setups[ig], INS.psolver_spectral(setups[ig]), closure, st)

# Define params where we want to evaluate the model
dt = params.Δt
tspan = [T(0), T(1)]

u0 = INS.random_field(setups[ig], T(0))
prob = ODEProblem(dudt_nn, u0, tspan, θ_posteriori)
sol = solve(prob, Tsit5(); u0 = u0, p = θ_posteriori, dt = dt, adaptive = false)

# Plot the field results
function animation_plots(sol_ode, setup; variable = "vorticity")
    anim = Plots.Animation()
    for (idx, (t, u)) in enumerate(zip(sol_ode.t, sol_ode.u))
        if variable == "vorticity"
            ω = INS.vorticity(u, setup)
            title = @sprintf("Vorticity, t = %.3f s", t)
            vor_lims = extrema(ω) # can specify manually other limits
            fig = Plots.heatmap(ω'; xlabel = "x", ylabel = "y", title, clims = vor_lims,
                color = :viridis, aspect_ratio = :equal, ticks = false, size = (600, 600))
        else
            title = @sprintf("\$u\$, t = %.3f s", t)
            fig = Plots.heatmap(u[:, :, 1]; xlabel = "x", ylabel = "y", title,
                aspect_ratio = :equal, ticks = false, size = (600, 600))
        end
        Plots.frame(anim, fig)
    end
    if variable == "vorticity"
        Plots.gif(anim, "simulations/NavierStokes_2D/plots/vorticity.gif", fps = 15)
    else
        Plots.gif(anim, "simulations/NavierStokes_2D/plots/velocity.gif", fps = 15)
    end
end

animation_plots(sol, setups[ig]; variable = "vorticity")
animation_plots(sol, setups[ig]; variable = "velocity")
