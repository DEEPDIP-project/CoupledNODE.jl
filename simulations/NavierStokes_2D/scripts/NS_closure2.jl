using IncompressibleNavierStokes
INS = IncompressibleNavierStokes


# Setup and initial condition
T = Float32
ArrayType = Array
Re = T(1_000)
n = 64
# this is the size of the domain, do not mix it with the time
lims = T(0), T(1);
x , y = LinRange(lims..., n + 1), LinRange(lims..., n + 1);
setup = INS.Setup(x, y; Re, ArrayType);
ustart = INS.random_field(setup, T(0));

dt = T(1e-3);
trange = [T(0), T(2e-3)]
saveat = [dt, 2dt];
u0stacked = stack(ustart)
N=n+2
u0 = reshape(u0stacked, N*N,2)
u0 = u0stacked

# ------ Use Lux to create a dummy_NN
import Random, Lux;
Random.seed!(123);
rng = Random.default_rng();

#dummy_NN = Lux.Chain(
#    x -> view(x, :),
#    Lux.Dense(N*N*2=>N*N*2, tanh),
#)
dummy_NN = Lux.Chain(
    Lux.ReshapeLayer((N,N,1)),
    Lux.Conv((3, 3), 1 => 1, pad=(1, 1)),
    x -> view(x, :),  # Flatten the output
)
θ0, st0 = Lux.setup(rng, dummy_NN)
st_node = st0

using ComponentArrays
θ_node = ComponentArray(θ0)
temp = similar(u0)

view(temp,:) .= Lux.apply(dummy_NN, u0, θ_node , st_node)[1]
temp == u0


# Force+NN in-place version
dudt_nn(du, u, P, t) = begin 
    view(du, :) .= Lux.apply(dummy_NN, u, P , st_node)[1]
    nothing
end

P = θ_node
temp = similar(u0);
dudt_nn(temp, u0, P, 0.0f0)
using OrdinaryDiffEq
prob_node = ODEProblem{true}(dudt_nn, u0, trange, p=P)

sol_node, time_node, allocation_node, gc_node, memory_counters_node = @timed Array(solve(prob_node, RK4(), u0 = u0, p = P, saveat = saveat))




########################
# Test the autodiff using Enzyme 
using Enzyme
using ComponentArrays
using SciMLSensitivity





# Define a posteriori loss function that calls the ODE solver
# First, make a shorter run
# and remember to set a small dt
prob = ODEProblem{true}(dudt_nn, u0stacked, trange, p=P)
ode_data = Array(solve(prob, RK4(), u0 = u0, p = P, saveat = saveat))
ode_data += T(0.5)*rand(Float32, size(ode_data))

myprob = ODEProblem{true}(dudt_nn, u0, trange, p=P)
pred = Array(solve(myprob, RK4(), u0 = u0, p = P, saveat=saveat))
P
sum(abs2,pred-ode_data)

# the loss has to be in place 
function loss(l::Vector{Float32},P, u0::Array{Float32}, tspan::Vector{Float32}, t::Vector{Float32})
    myprob = ODEProblem{true}(dudt_nn, u0, tspan, P)
    pred = Array(solve(myprob, RK4(), u0 = u0, p = P, saveat=t))
    l .= Float32(sum(abs2, ode_data- pred))
    nothing
end

l=[T(0.0)];
loss(l,P, u0, trange, saveat)
l


# Test if the loss can be autodiffed
# [!] dl is called the 'seed' and it has to be marked to be one for correct gradient
l = [T(0.0)];
dl = Enzyme.make_zero(l) .+T(1);
dP = Enzyme.make_zero(P);
du = Enzyme.make_zero(u0);
@timed Enzyme.autodiff(Enzyme.Reverse, loss, Duplicated(l, dl), Duplicated(P, dP), DuplicatedNoNeed(u0, du), Const(trange), Const(saveat))
dP
    
    