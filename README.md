# SciML-model-coupling

### Context

This repo was created during [this co-working session](https://github.com/DEEPDIP-project/logs/blob/main/meetings/2024-02-20%20Coworking%20session.md).

----

## C-NODEs in SciML

To motivate the usefulness of this repo, we will start answering two questions:
* (1) **What** are NODEs? 
* (2) **What** are C-NODEs?
* (2) **Why** do we want C-NODEs?

### (1) What are NODEs?

Let's consider a generic differential equation (DE):

$$
\begin{cases} \frac{du}{dt} = f(t,u) \\
u(t=0) = u_0 \end{cases} \tag{1.0}
$$

where we look for the time-dependent solution $u(t)$, given its initial condition $u_0$ and the force $f(t,u)$ acting on it.
In general, we have an ordinary differential equation (ODE) when $$\frac{du}{dt} = f(u, t)$$ contains a single independent variable $u$.

We have a Neural ODE (NODE) when the structure of the problem becomes: $$\frac{du}{dt} = f(u,t) + NN(u |\theta),$$ so there is an extra force terms that depends on extra parameters $\theta$ that are not part of the solution. This term is called $NN$ because it can be a neural network. Overall, a NODE problem is a data-driven method that aims to model the solution $u(t)$ when the ODE can not be solved exactly, or even if $f$ is unknown. Let's look at the details with an example.

#### Example 01.00: Logistic equation

* [Example 01.00](examples/01.00-Logistic.jl)

### (2) What are C-NODEs?

Coupled Neural ODEs (C-NODEs) are defined as:

$$
\begin{pmatrix} \dfrac{du}{dt}\\ \dfrac{dv}{dt} \end{pmatrix} = \begin{pmatrix} f(u,v,t)\\g(u,v,t) \end{pmatrix} + \begin{pmatrix}NN_f(u,v|\theta_f) \\NN_g(u,v|\theta_g) \end{pmatrix}.
$$

Let's check with an example how this works.

#### Example 02.00: Gray-Scott model
* [Example 02.00](examples/02.00-GrayScott.jl)

### How can we train a CNODE?
A priori fitting
#### Example 02.01: Learning the Gray-Scott model
* [Example 02.01](examples/02.01-GrayScott.jl)
  
A posteriori fitting
#### Example 02.02: Learning the Gray-Scott model
* [Examples 02.02](examples/02.01-GrayScott_wMultDTO.jl)

### (3) Why coupled NODEs? 
Two main explanations:
* We are working with coupled systems (like Gray-Scott).
* Multi-scale approach to PDEs.

#### Example 2.1: Parameters learning of the Gray-Scott problem
* Link: Gray-Scott parameter learning
  
#### Example 2.2: Gray-Scott as a multiscale problem
* Link: Gray-Scott multiscale

### Multiscale approach to chaotic PDEs
In a more realistic situation, the incognita is usually a vector field, so we will try to solve for $u(x,t): \Omega \times \mathbb{R} \rightarrow \mathbb{R}^m$, where the spacial coordinate is $x\in \Omega \subseteq \mathbb{R}^D$ and the field $u$ is $m$-dimensional.

If the problem is simple, like the case of linear ODEs, $u$ will be perfectly representable using a few parameters, e.g phase and frequency of an harmonic oscillator, and an analytical solution will be probably available.
If this is the case, one could rely on direct and efficient approaches like the one described [here](https://docs.sciml.ai/DiffEqDocs/stable/examples/classical_physics/).

However most of the times, the analytical solution is not available. A common solution is the method of lines, that consists in discretizing the spacial dimension and study a vectorial representation of the function $u(x,t) \rightarrow \bar{u}(t)\in \mathbb{R}^N$, where $N\ll D$. This discretization can also be interpreted as a **filter** $\Phi\in\mathbb{R}^{D\times N}$ that gets applied to the field:
$$\bar{u}= \Phi u$$

Once we have a discretized space, we also discretize the force $f\rightarrow f_h$ using finite differences, while also assuming that the force is not time dependent. So we are left with

$$
\tag{2.2.1}
\begin{cases}
\frac{d\bar{u}}{dt} = f_h(\bar{u}) \\
\bar{u}(t=0) = \Phi u(t=0) = \bar{u}_0
\end{cases}
$$

Then the crucial question that we need to address is 

$$
\tag{2.2.2}\bar{u}(t)\stackrel{?}{=} \Phi u(t),
$$

or in words: **does the discretized solution evolve like the original?**

The answer is that it depends on the harshness of the discretization, i.e. the value of $N$ for the specific problem.
We distinguish two cases:
* **DNS** (direct numerical simulation): if $N$ is large then eq.\ref{eq:isclosed} is true, or we can say that the system is effectively *closed* because the discretized solution evolves following the behavior of the original system.
However, especially if the system is chaotic, it is likely that the small scale behavior influences the large scale behavior. In this case we talk about:
* **LES** (large eddy simulations): $N$ is too small to capture the small scale effects, thus the eq.\ref{eq:ode} is *closed* because we would need $$\frac{d\bar{u}}{dt} = \Phi f(u)$$ which can not be solved using $\bar{u}(t)$ only. 

... TO BE CONTINUED ...

We are mainly interested in differential equation models for vector fields that have some type of chaotic behavior. This means that we can expect that the problem can not be solved for a continuous variable $x\in\mathbb{R}$, so the solution has to be discretized on a finite grid. The definition of this grid also influences the definition of the force $f$, because usually the derivatives are approximated into finite differences. It is then important to consider that the grid is strictly connected to the boundary conditions: if $x=0$ is a fixed boundary, the staggered grid $x=n/2$ for $n=1,\cdots,N$ would miss this boundary. 
For now, **we leave the grid problem to the user**. This means that we assume that the DE and all its elements will be defined by the user on a grid that makes sense for its problem, and we will **assume this constant user-defined grid for the solution** of the DE.

### Example 4 : Burgers
Look [here](https://github.com/DEEPDIP-project/NeuralNS-SciML-Tutorials). It has to be adapted, generalized and moved to this repo.

### Example 5 : Navier-Stokes
It is [here](https://github.com/DEEPDIP-project/NeuralNS-SciML-Tutorials). Make it compatible with this repo.

## References

* [SciML style guidelines](https://github.com/SciML/SciMLStyle)
