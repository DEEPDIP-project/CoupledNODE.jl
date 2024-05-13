# # Logistic equation and NODE
# Let's study a phenomenon that can be described with the following ODE $$\dfrac{dP}{dt} = rP\left(1-\dfrac{P}{K}\right),$$ which is called the logistic equation. Given $P(t=0)=P_0$ we can solve this problem analytically to get $P(t) = \frac{K}{1+\left(K-P_0\right)/P_0 \cdot e^{-rt}}$. Let's plot the solution for $r=K=2, P_0=0.01$:
r = 2
K = 1
P0 = 0.01
function P(t)
    return K / (1 + (K - P0) * exp(-r * t) / P0)
end
t = range(start = 0, stop = 6, step = 0.01)
Pt = P.(t)
using Plots
plot(t, Pt, label = "P(t)", xlabel = "t", ylabel = "P(t)")

# Let's say that we want to use the logistic equation to model an experiment like the activation energy of a neuron. We run the experiment and we observe the following:
import DifferentialEquations: ODEProblem, solve, Tsit5
function observation()
    f_o(u, p, t) = u .* (0.0 .- 0.8 .* log.(u))
    trange = (0.0, 6.0)
    prob = ODEProblem(f_o, 0.01, trange, dt = 0.01, saveat = 0.01)
    sol = solve(prob, Tsit5())
    return sol.u
end
u_experiment = observation()
using Plots
plot(t, Pt, label = "Best P(t) fit")
plot!(t, u_experiment[:], label = "Observation(t)")

# This means that our experimental system, despite its similarity, it is not described by a logistic ODE.
# How can we then model our experiment as an ODE?
# 
# There are many simpler alternatives for this example (e.g. Mode Decomposition, SINDY or Bayesian methods), but let's use this exercise to introduce a NODE:   
# $\begin{equation}\dfrac{du}{dt} = \underbrace{ru\left(1-\dfrac{u}{K}\right)}_{f(u)} + NN(u|\theta).\end{equation}$
# In this NODE we are looking for a solution $u(t)$ that reproduces our observation.
# We will be using [SciML](https://sciml.ai/) package [DiffEqFlux.jl](https://github.com/SciML/DiffEqFlux.jl) and scpecifically [NeuralODE](https://docs.sciml.ai/DiffEqFlux/stable/examples/neural_ode/) for defining and solving the problem.
# ## Solve the NODE
# We solve this 1D NODE using introducing the functionalities of this repository:

# * We create the NN using `Lux`. In this example we do not discuss the structure of the NN, but we leave it as a black box that can be designed by the user. We will show later how to take inspiration from the physics of the problem to design optimal NN.
import Lux
NN = Lux.Chain(Lux.SkipConnection(Lux.Dense(1, 3),
    (out, u) -> u .* out[1] .+ u .* u .* out[2] .+ u .* log.(abs.(u)) .* out[3]));
# * We define the force $f(u)$ compatibly with SciML. 
f_u(u) = @. r * u * (1.0 - u / K);

# * We create the right hand side of the NODE, by combining the NN with f_u
import CoupledNODE: create_f_CNODE, Grid, linear_to_grid
grid_u = Grid(dim = 1, dx = 1.0, nx = 1)
f_NODE = create_f_CNODE((f_u,), (grid_u,), (NN,); is_closed = true);

# and get the parametrs that you want to train
import Random
rng = Random.seed!(1234)
θ, st = Lux.setup(rng, f_NODE);

# * We define the NODE
import DiffEqFlux: NeuralODE
trange = (0.0, 6.0)
u0 = hcat([0.01]) # need the inputs as Matrix instead of Vector
full_NODE = NeuralODE(f_NODE, trange, Tsit5(), adaptive = false, dt = 0.001, saveat = 0.2);

# * We solve the NODE using the zero-initialized parameters
# *Note:* `full_NODE` is a `NeuralODE` function that returns a `DiffEqBase.ODESolution` object. This object contains the solution of the ODE, but it also contains additional information like the time points at which the solution was evaluated and the parameters of the ODE. We can access the solution using `[1]`, and we convert it to an `Array` to be able to use it for further calculations and plot.
untrained_NODE_solution = Array(full_NODE(u0, θ, st)[1]);

# ## Prepare the model
# First, we define this auxiliary NODE that will be used for training
dt = 0.01 # it has to be as fine as the data
nunroll = 60
t_train_range = (0.0, dt * nunroll) # it has to be as long as unroll
training_NODE = NeuralODE(f_NODE,
    t_train_range,
    Tsit5(),
    adaptive = false,
    dt = dt,
    saveat = dt);

# We reshape the data to be compatible with the loss function. We want a vector with shape (dim_u, n_samples, t_steps)
u_experiment_mod = reshape(u_experiment, grid_u.nx, 1, length(u_experiment))
# Second, we need to design the **loss function**. For this example, we use *multishooting a posteriori* fitting [(MulDtO)](https://docs.sciml.ai/DiffEqFlux/dev/examples/multiple_shooting/). Using `Zygote` we compare `nintervals` of length `nunroll` to get the gradient. Notice that this method is differentiating through the solution of the NODE!
import CoupledNODE: create_randloss_MulDtO
nintervals = 5
myloss = create_randloss_MulDtO(
    u_experiment_mod, training_NODE, st, nunroll = nunroll, nintervals = nintervals, nsamples = 1);

# Initialize and trigger the compilation of the model
import ComponentArrays
pinit = ComponentArrays.ComponentArray(θ);
myloss(pinit); # trigger compilation

# Select the autodifferentiation type
import OptimizationOptimisers: Optimization
adtype = Optimization.AutoZygote();
# We transform the NeuralODE into an optimization problem
optf = Optimization.OptimizationFunction((x, p) -> myloss(x), adtype);
optprob = Optimization.OptimizationProblem(optf, pinit);

# Select the training algorithm:
# Adam with learning rate 0.01, with gradient clipping
import OptimizationOptimisers: OptimiserChain, Adam, ClipGrad
algo = OptimiserChain(Adam(1.0e-2), ClipGrad(1));

# Or this other optimizer (uncomment tu use).
#import OptimizationOptimJL: Optim
#algo = Optim.LBFGS();

# ## Train de NODE
# We are ready to train the NODE.
# Notice that the block can be repeated to continue training
import CoupledNODE: callback
result_neuralode = Optimization.solve(optprob,
    algo;
    callback = callback,
    maxiters = 1000)
pinit = result_neuralode.u;
optprob = Optimization.OptimizationProblem(optf, pinit);

# ## Analyse the results
# Visualize the obtained results
plot()
plot(t, Pt, label = "Best P(t) fit")
plot!(t, u_experiment[:], label = "Observation(t)")
scatter!(range(start = 0, stop = 6, step = 0.2),
    untrained_NODE_solution[:],
    label = "untrained NODE",
    marker = :circle)
scatter!(range(start = 0, stop = 6, step = 0.2),
    Array(full_NODE(hcat([u_experiment[1]]), result_neuralode.u, st)[1])[:],
    label = "Trained NODE",
    marker = :circle)
