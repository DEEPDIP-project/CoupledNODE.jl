using OrdinaryDiffEq, SciMLSensitivity, Optimization, OptimizationOptimisers,
      Plots

u0 = [2.0; 0.0]
datasize = 30
tspan = (0.0, 1.5)
datasize = 10
tspan = (0.0, 0.5)

function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u .^ 3)'true_A)'
end
t = range(tspan[1], tspan[2], length = datasize)
typeof(t)
prob = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob, Tsit5(), saveat = t))

#dudt2 = Flux.Chain(x -> x .^ 3,
#    Flux.Dense(2, 50, tanh),
#    Flux.Dense(50, 2)) |> f64
#p, re = Flux.destructure(dudt2) # use this p as the initial condition!
#dudt(u, p, t) = re(p)(u) # need to restrcture for backprop!

import Random, Lux;
Random.seed!(123);
rng = Random.default_rng();
dudt2= Lux.Chain(
    x -> x .^ 3,
    Lux.Dense(2, 50, tanh),
    Lux.Dense(50, 2)
)
p, st = Lux.setup(rng, dudt2)
using ComponentArrays
p = ComponentArray(p)
dudt(du, u, p, t) = begin
    du .= Lux.apply(dudt2, u, p, st)[1]
    nothing
end
u0
p
du = similar(u0)
dudt(du, u0, p, 0.0)

du
prob = ODEProblem(dudt, u0, tspan, p)

Array(solve(prob, Tsit5(), u0 = u0, p = p, saveat = t, dt=0.01))

pred = similar(ode_data)
function loss_n_ode(θ)
    pred = Array(solve(prob, Tsit5(), u0 = u0, p = θ, saveat = t, sensealg=SciMLSensitivity.EnzymeVJP()))
    loss = sum(abs2, ode_data .- pred)
    loss, pred
end
function loss2(θ, u0, t, du)
    pred = solve(prob, RK4(), u0 = u0, p = θ, saveat = t, sensealg=SciMLSensitivity.EnzymeVJP(), dt=0.01)
#    AA = init(prob, RK4(), u0 = u0, p = θ, saveat = t, sensealg=SciMLSensitivity.EnzymeVJP())
#    pred = solve!(AA)
    return sum(abs2, ode_data .- pred[1])

#    # Remember to allocate also du as dual
#    pred = dudt(du, u0, θ, 0.0)
#    return sum(abs2,du)
end

loss_n_ode(p)
loss2(p,u0,t, du)
prob

# to check 
# https://docs.sciml.ai/DiffEqDocs/stable/basics/faq/#Autodifferentiation-and-Dual-Numbers
# https://enzyme.mit.edu/julia/stable/faq/#Activity-of-temporary-storage

using Enzyme
#Enzyme.API.runtimeActivity!(true)
#Enzyme.API.runtimeActivity!(false)
Enzyme.autodiff(Enzyme.Reverse, loss2, Active, Duplicated(p, Enzyme.make_zero(p)), Duplicated(u0, Enzyme.make_zero(u0)), Duplicated(t, Enzyme.make_zero(t)), Duplicated(Enzyme.make_zero(u0), Enzyme.make_zero(u0)))
#Enzyme.autodiff(Enzyme.Reverse, loss2, Active, Duplicated(p, Enzyme.make_zero(p)), Duplicated(u0, Enzyme.make_zero(u0)), Duplicated(t, Enzyme.make_zero(t)), Duplicated(pred, Enzyme.make_zero(pred)))
show(err)





callback = function (θ, l, pred; doplot = false) #callback function to observe training
    display(l)
    # plot current prediction against data
    pl = scatter(t, ode_data[1, :], label = "data")
    scatter!(pl, t, pred[1, :], label = "prediction")
    display(plot(pl))
    return false
end

# Display the ODE with the initial parameter values.
callback(p, loss_n_ode(p)...)

# use Optimization.jl to solve the problem
adtype = Optimization.AutoZygote()
using Enzyme
adtype = Optimization.AutoEnzyme()

optf = Optimization.OptimizationFunction((p, _) -> loss_n_ode(p), adtype)
optprob = Optimization.OptimizationProblem(optf, p)

result_neuralode = Optimization.solve(optprob,
    OptimizationOptimisers.Adam(0.05),
    callback = callback,
    maxiters = 300)
show(err)



using Enzyme
u0 = prob.u0
p = prob.p
tmp2 = Enzyme.make_zero(p)
t = prob.tspan[1]
du = zero(u0)

const dux = zero(u0)
if DiffEqBase.isinplace(prob)
    _f = prob.f
else
    _f = (du, u, p, t) -> (du .= prob.f(u, p, t); nothing)
end
_f

_tmp6 = Enzyme.make_zero(_f)
tmp3 = zero(u0)
tmp4 = zero(u0)
ytmp = zero(u0)
tmp1 = zero(u0)
tmpt = zero(t)

Enzyme.autodiff(Enzyme.Reverse, _f,
    Const, Duplicated(tmp3, tmp4),
    Duplicated(ytmp, tmp1),
    Duplicated(p, tmp2),
    Duplicated(t, tmpt))
tmp2
show(err)