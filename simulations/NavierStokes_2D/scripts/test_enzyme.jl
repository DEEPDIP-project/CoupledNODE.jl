using SimpleChains, StaticArrays, OrdinaryDiffEq, SciMLSensitivity, Optimization, OptimizationOptimisers, Plots

u0 = [2.0; 0.0]
du = similar(u0)
datasize = 30
tspan = [0.0, 1.5]


function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u .^ 3)'true_A)'
end
#const t = Array(range(tspan[1], tspan[2], length = datasize))
t = Array(range(tspan[1], tspan[2], length = datasize))
typeof(t)
prob0 = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob0, Tsit5(), saveat = t, sensealg = DiffEqBase.SensitivityADPassThrough()))


import Random
Random.seed!(123);
rng = Random.default_rng();


sc = SimpleChain(static(2),
    Activation(x -> x .^ 3),
    TurboDense{true}(tanh, static(50)),
    TurboDense{true}(identity, static(2)))
p_nn = Array{Float64}(SimpleChains.init_params(sc)) 
function dudt(du::Vector{Float64}, u::Vector{Float64}, p::Vector{Float64}, t::Float64) 
    du .= sc(u, p)
    nothing
end
dudt(du, u0, p_nn, 0.0)

prob = ODEProblem{true}(dudt, u0, tspan, p_nn)

Array(DiffEqBase.solve(prob, Tsit5(), u0 = u0, p = p_nn, saveat = t))

pred = similar(ode_data)

#fix this loss to make it good as the other (one input only)
function loss_n_ode(θ::Vector{Float64})
    myprob = ODEProblem{true}(dudt, u0, tspan, θ)
    # Remember to not use Enzyme sensealg with Zygote
    # with SimpleChains it has problems in selecting sensealg
    pred = Array(solve(myprob, RK4(), u0 = u0, p = θ, saveat=t, verbose=false))
    return sum(abs2, ode_data - pred), pred
end
function loss_mock(θ::Vector{Float64})
    return 0.0, ode_data.+1 
end

function lop(θ::Vector{Float64}, u0::Vector{Float64}, tspan::Vector{Float64}, t::Vector{Float64})
    myprob = ODEProblem{true}(dudt, u0, tspan, θ)
    # using verbose=false to avoid printing the warning about AD
    # the warning says that sensealg is not optimal but we do not care since we do AD externally
    pred = Array(solve(myprob, RK4(), u0 = u0, p = θ, saveat=t, verbose=false))
    return sum(abs2, ode_data - pred)
end
function lip(l::Vector{Float64},θ::Vector{Float64}, u0::Vector{Float64}, tspan::Vector{Float64}, t::Vector{Float64})
    myprob = ODEProblem{true}(dudt, u0, tspan, θ)
    pred = Array(solve(myprob, RK4(), u0 = u0, p = θ, saveat=t, verbose=false))
    l .= Float64(sum(abs2, ode_data - pred))
    nothing
end

loss_n_ode(p_nn)
lop(p_nn,u0,tspan, t)
l=[0.0]
lip(l,p_nn,u0,tspan, t)


using Enzyme
# They store the gradients
dp = Enzyme.make_zero(p_nn) .+ 1
du0 = Enzyme.make_zero(u0)
dts = Enzyme.make_zero(tspan)
Enzyme.autodiff(Enzyme.Reverse, lop, Active, DuplicatedNoNeed(p_nn, dp), DuplicatedNoNeed(u0, du0) , Const(tspan), Const(t))
# [!] dl is called the 'seed' and it has to be marked to be one for correct gradient
l = [0.0]
dl = Enzyme.make_zero(l) .+1
Enzyme.autodiff(Enzyme.Reverse, lip, Duplicated(l, dl), Duplicated(p_nn, dp), DuplicatedNoNeed(u0, du0) , Const(tspan), Const(t))


function my_G(G, θ::Vector{Float64}, u0_tspan_t_dp_du0::Vector{Vector{Float64}}) 
    u0, tspan, t, du0 = u0_tspan_t_dp_du0
    # Reset gradient to zero
    Enzyme.make_zero!(G)
    # And remember to pass the seed to the loss funciton with the dual part set to 1
    Enzyme.autodiff(Enzyme.Reverse, lip, Duplicated([0.0], [1.0]), Duplicated(θ, G), DuplicatedNoNeed(u0, du0) , Const(tspan), Const(t))
    nothing
end

G = copy(dp)
oo = my_G(G,p_nn, [u0, tspan, t, du0])


callback = function (θ, l, pred; doplot = false) #callback function to observe training
    display(l)
    # plot current prediction against data
    pl = scatter(t, ode_data[1, :], label = "data")
    scatter!(pl, t, pred[1, :], label = "prediction")
    display(plot(pl))
    return false
end

# Display the ODE with the initial parameter values.
callback(p_nn, loss_n_ode(p_nn)...)

# use Optimization.jl to solve the problem
# First solve using Zygote
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((p,_) -> loss_n_ode(p), adtype)
optprob = Optimization.OptimizationProblem(optf, p_nn)

result_z, time_z, alloc_z, gc_z, mem_z = @timed Optimization.solve(optprob,
    OptimizationOptimisers.Adam(0.05),
    callback = callback,
    maxiters = 100)

# and now solve using Enzyme
optf = Optimization.OptimizationFunction((p,_)->loss_n_ode(p), grad=(G,p,u)->my_G(G,p,u))
optprob = Optimization.OptimizationProblem(optf, p_nn, [u0, tspan, t, du0])

result_e, time_e, alloc_e, gc_e, mem_e = @timed Optimization.solve(optprob,
    OptimizationOptimisers.Adam(0.05),
    callback = callback,
    maxiters = 100)

# plot the comparison
using Plots
p1 = Plots.bar(
    ["Zygote", "Enzyme"], [time_z, time_e], xlabel = "Method",
    ylabel = "Time (s)", title = "Time comparison", legend = false);
p2 = Plots.bar(["Zygote", "Enzyme"],
    [mem_z.allocd, mem_e.allocd],
    xlabel = "Method", ylabel = "Memory (bytes)",
    title = "Memory comparison", legend = false);
p3 = Plots.bar(["Zygote", "Enzyme"], [gc_z, gc_e], xlabel = "Method",
    ylabel = "Number of GC", title = "GC comparison", legend = false);
p4 = Plots.bar(["Zygote", "Enzyme"], [result_z.objective, result_e.objective], xlabel = "Method",
    ylabel = "Final loss", title = "Loss comparison", legend = false);
# make a single plot
plot(p1, p2, p3, p4, layout = (2, 2), size = (800, 800))

# Compute the two different solutions
myprob = ODEProblem{true}(dudt, u0, tspan, result_z.u)
pred_z = Array(solve(myprob, RK4(), u0 = u0, p = result_z.u, saveat=t, verbose=false))
myprob = ODEProblem{true}(dudt, u0, tspan, result_e.u)
pred_e = Array(solve(myprob, RK4(), u0 = u0, p = result_e.u, saveat=t, verbose=false))
plot(t, ode_data[1, :], label = "data")
plot!(t, pred_z[1, :], label = "Zygote")
plot!(t, pred_e[1, :], label = "Enzyme")

# Are they identical?
plot(t, pred_e[1, :].-pred_z[1, :])