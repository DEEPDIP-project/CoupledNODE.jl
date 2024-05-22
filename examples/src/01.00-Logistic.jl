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

# * We create a spatial grid. In this case we have a 1D problem with a single point.
import CoupledNODE: create_f_CNODE, Grid
grid_u = Grid(dim = 1, dx = 1.0, nx = 1)

# ## Prepare the model
# We define parameters necessary for the solution such as time step and how often to save the solutions.
dt = 0.01 # it has to be as fine as the data
nunroll = 60
t_range = (0.0, 6.0) # it has to be as long as unroll

# We define our model `NeuralODE` with all the elements that we have defined until now.
full_NODE = create_f_CNODE((f_u,), (grid_u,), t_range, Tsit5(), (NN,); adaptive = false,
    dt = 0.001, saveat = 0.2, is_closed = true);

# and get the parameters that you want to train
import Random
rng = Random.seed!(1234)
θ, st = Lux.setup(rng, full_NODE.model);

# * We solve the NODE using the zero-initialized parameters to get a reference solution (an untrained one).
# *Note:* `full_NODE` is a `NeuralODE` function that returns a `DiffEqBase.ODESolution` object. This object contains the solution of the ODE, but it also contains additional information like the time points at which the solution was evaluated and the parameters of the ODE. We can access the solution using `[1]`, and we convert it to an `Array` to be able to use it for further calculations and plot. In this case we get the untrained solution for reference.
untrained_NODE_solution = Array(full_NODE(hcat([u_experiment[1]]), θ, st)[1]);
# We reshape the data (labels) to be compatible with the loss function. We want a vector with shape (dim_u, n_samples, t_steps)
u_experiment_mod = reshape(u_experiment, grid_u.nx, 1, length(u_experiment))

# We create an auxiliary `training_NODE` thatis in principle the same as `full_NODE`, but it is used to train the NODE with different time steps and different number of samples.
dt = 0.01 # it has to be as fine as the data
nunroll = 60
t_train_range = (0.0, dt * nunroll) # it has to be as long as unroll
training_NODE = create_f_CNODE((f_u,), (grid_u,), t_train_range, Tsit5(), (NN,);
    adaptive = false, dt = dt, saveat = dt, is_closed = true);

# We design the **loss function**. For this example, we use *multishooting a posteriori* fitting [(MulDtO)](https://docs.sciml.ai/DiffEqFlux/dev/examples/multiple_shooting/). Using `Zygote` we compare `nintervals` of length `nunroll` to get the gradient. Notice that this method is differentiating through the solution of the NODE!
import CoupledNODE: create_randloss_MulDtO
myloss = create_randloss_MulDtO(
    u_experiment_mod, training_NODE, st, nunroll = nunroll, nintervals = 5, nsamples = 1);

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
result_training = Optimization.solve(optprob,
    algo;
    callback = callback,
    maxiters = 1000)
pinit = result_training.u; #optimized params
optprob = Optimization.OptimizationProblem(optf, pinit);

# And we get our results, notice that we use `full_NODE` instead of `training_NODE` so that it will plot as expected. Also notice that we use the optimized parameters `pinit`. 
trained_NODE_solution = Array(full_NODE(hcat([u_experiment[1]]), pinit, st)[1]);

# ## Analyse the results
# Visualize the obtained results
plot(t, Pt, label = "Best P(t) fit")
plot!(t, u_experiment[:], label = "Observation(t)")
scatter!(range(start = 0, stop = 6, step = 0.2),
    untrained_NODE_solution[:],
    label = "untrained NODE",
    marker = :circle)
scatter!(range(start = 0, stop = 6, step = 0.2),
    trained_NODE_solution[:],
    label = "Trained NODE",
    marker = :circle)
