using StaticArrays, OrdinaryDiffEq, SciMLSensitivity, Optimization, OptimizationOptimisers, Plots


u0 = [2.0; 0.0]
du = similar(u0)
datasize = 30
tspan = [0.0, 1.5]

function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u .^ 3)'true_A)'
end
t = Array(range(tspan[1], tspan[2], length = datasize))
typeof(t)
prob0 = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob0, Tsit5(), saveat = t, sensealg = DiffEqBase.SensitivityADPassThrough()))

import Random
Random.seed!(123);
rng = Random.default_rng();


# Force using Lux
using Lux
lc= Lux.Chain(
    x -> x .^ 3,
    ReshapeLayer((1,1)),
    x -> reshape(x, 2),
    Lux.Dense(2, 50, tanh),
    Lux.Dense(50, 2)
)
p_nn, st = Lux.setup(rng, lc)
using ComponentArrays
p_nn = ComponentArray(p_nn)
dudt_l(du, u, p, t) = begin
    du .= Lux.apply(lc, u, p, st)[1]
    nothing
end
dudt_l(du, u0, p_nn, 0.0)




prob = ODEProblem{true}(dudt_l, u0, tspan, p_nn)
sol_l = Array(solve(prob, Tsit5(), u0 = u0, p = p_nn, saveat = t))
using Plots
plot(t, ode_data[1, :], label = "data")
plot!(t, sol_l[1, :], label = "Lux t=0")



# Define the loss funcitons
function loss_l(θ)
    myprob = ODEProblem{true}(dudt_l, u0, tspan, θ)
    pred = Array(solve(myprob, RK4(), u0 = u0, p = θ, saveat=t, verbose=false))
    return sum(abs2, ode_data - pred), pred
end
# This in place version is needed for Enzyme
function lip_l(l::Vector{Float64},θ, u0::Vector{Float64}, tspan::Vector{Float64}, t::Vector{Float64})
    myprob = ODEProblem{true}(dudt_l, u0, tspan, θ)
    pred = Array(solve(myprob, RK4(), u0 = u0, p = θ, saveat=t, verbose=false))
    l .= Float64(sum(abs2, ode_data - pred))
    nothing
end

# trigger the compilation
loss_l(p_nn)
l=[0.0]
lip_l(l,p_nn,u0,tspan, t)
l


callback = function (θ, l, pred; doplot = false) #callback function to observe training
    display(l)
    # plot current prediction against data
    pl = scatter(t, ode_data[1, :], label = "data")
    scatter!(pl, t, pred[1, :], label = "prediction")
    display(plot(pl))
    return false
end
callback(p_nn, loss_l(p_nn)...)



# ******** Enzyme + Lux ********
using Enzyme
dp = Enzyme.make_zero(p_nn)
du0 = Enzyme.make_zero(u0)
function G_l!(G, θ, u0_tspan_t_dp_du0::Vector{Vector{Float64}}) 
    u0, tspan, t, du0 = u0_tspan_t_dp_du0
    Enzyme.make_zero!(G)
    Enzyme.autodiff(Enzyme.Reverse, lip_l, Duplicated([0.0], [1.0]), Duplicated(θ, G), DuplicatedNoNeed(u0, du0) , Const(tspan), Const(t))
    nothing
end
G_l!(copy(dp),p_nn, [u0, tspan, t, du0])

optf = Optimization.OptimizationFunction((p,_)->loss_l(p), grad=(G,p,u)->G_l!(G,p,u))
optprob = Optimization.OptimizationProblem(optf, p_nn, [u0, tspan, t, du0])

result_e_l, time_e_l, alloc_e_l, gc_e_l, mem_e_l = @timed Optimization.solve(optprob,
    OptimizationOptimisers.Adam(0.05),
    callback = callback,
    maxiters = 100)



# ******** Compare the results ********
using Plots
p1 = Plots.bar(
    ["Zygote\n+\nSimpleChain", "Zygote\n+\nLux", "Enzyme\n+\nSimpleChain", "Enzyme\n+\nLux"],
    [time_z_s, time_z_l, time_e_s, time_e_l], xlabel = "Method",
    ylabel = "Time (s)", title = "Time comparison", legend = false);
p2 = Plots.bar(["Zygote\n+\nSimpleChain", "Zygote\n+\nLux", "Enzyme\n+\nSimpleChain", "Enzyme\n+\nLux"],
    [mem_z_s.allocd, mem_z_l.allocd, mem_e_s.allocd, mem_e_l.allocd],
    xlabel = "Method", ylabel = "Memory (bytes)",
    title = "Memory comparison", legend = false);
p3 = Plots.bar(["Zygote\n+\nSimpleChain", "Zygote\n+\nLux", "Enzyme\n+\nSimpleChain", "Enzyme\n+\nLux"],
    [gc_z_s, gc_z_l, gc_e_s, gc_e_l], xlabel = "Method",
    ylabel = "Number of GC", title = "GC comparison", legend = false);
p4 = Plots.bar(["Zygote\n+\nSimpleChain", "Zygote\n+\nLux", "Enzyme\n+\nSimpleChain", "Enzyme\n+\nLux"],
    [result_z_s.objective, result_z_l.objective, result_e_s.objective, result_e_l.objective], xlabel = "Method",
    ylabel = "Final loss", title = "Loss comparison", legend = false);
# make a single plot
plot(p1, p2, p3, p4, layout = (2, 2), size = (800, 800))

# Compute the different solutions
myprob = ODEProblem{true}(dudt_sc, u0, tspan, result_z_s.u)
pred_z_s = Array(solve(myprob, RK4(), u0 = u0, p = result_z_s.u, saveat=t, verbose=false))
myprob = ODEProblem{true}(dudt_l, u0, tspan, result_z_l.u)
pred_z_l = Array(solve(myprob, RK4(), u0 = u0, p = result_z_l.u, saveat=t, verbose=false))
myprob = ODEProblem{true}(dudt_sc, u0, tspan, result_e_s.u)
pred_e_s = Array(solve(myprob, RK4(), u0 = u0, p = result_e_s.u, saveat=t, verbose=false))
myprob = ODEProblem{true}(dudt_l, u0, tspan, result_e_l.u)
pred_e_l = Array(solve(myprob, RK4(), u0 = u0, p = result_e_l.u, saveat=t, verbose=false))
plot(t, ode_data[1, :], label = "data")
plot!(t, pred_z_s[1, :], label = "Zygote + SimpleChain")
plot!(t, pred_z_l[1, :], label = "Zygote + Lux")
plot!(t, pred_e_s[1, :], label = "Enzyme + SimpleChain")
plot!(t, pred_e_l[1, :], label = "Enzyme + Lux")


# Are they identical?
plot(t, pred_z_s[1, :].-pred_z_l[1, :])
plot(t, pred_e_s[1, :].-pred_e_l[1, :])
plot(t, pred_z_s[1, :].-pred_e_s[1, :])
plot(t, pred_z_l[1, :].-pred_e_l[1, :])

