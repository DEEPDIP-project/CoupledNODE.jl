# CoupledNODE

[![Build Status](https://github.com/DEEPDIP-project/CoupledNODE/actions/workflows/CI.yml/badge.svg)](https://github.com/DEEPDIP-project/CoupledNODE/actions/workflows/CI.yml)

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

We have a Neural ODE (NODE) when the structure of the problem becomes: $$\frac{du}{dt} = f(u,t) + NN(u |\theta).$$ Notice the extra force term, named $NN$ because it can be a neural network, that depends on extra parameters $\theta$ that are not part of the solution. Overall, a NODE problem is a data-driven method that aims to model the solution $u(t)$ when the ODE can not be solved exactly, or even if $f$ is unknown. Let's look at the details with an example.

#### Example 01.00: Logistic equation

* [Example 01.00](examples/01.00-Logistic.jl)

### (2) What are C-NODEs?

Coupled Neural ODEs (C-NODEs) are defined as:

$$
\tag{2.1.0}
\begin{pmatrix} \dfrac{du}{dt} \\
\dfrac{dv}{dt} \end{pmatrix} = \begin{pmatrix} f(u,v,t)\\
g(u,v,t) \end{pmatrix} + \begin{pmatrix}NN_f(u,v|\theta_f) \\
NN_g(u,v|\theta_g) \end{pmatrix}.
$$

Let's check with an example of the Gray-Scott model how such C-NODEs could be used.
In the following, we are going to:

1. [Example 02.00](examples/02.00-GrayScott.jl): Introduce and solve the  model using a explicit solver.
2. [Example 02.01](examples/02.01-GrayScott.jl): Introduce a Neural operator, train it using **a priori fitting** and solve the Gray-Scott model using it.
3. [Examples 02.02](examples/02.02-GrayScott.jl): Train another Neural operator using **a posteriori fitting** and solve the Gray-Scott model using it.

### (3) Why coupled NODEs? 
Two main explanations:
* We are working with coupled systems (like Gray-Scott).
* Multi-scale approach to PDEs.


### Multiscale approach to chaotic PDEs
In a more realistic situation, the incognita is usually a vector field, so we will try to solve for $u(x,t): \Omega \times \mathbb{R} \rightarrow \mathbb{R}^m$, where the spacial coordinate is $x\in \Omega \subseteq \mathbb{R}^D$ and the field $u$ is $m$-dimensional.

If the problem is simple, like the case of linear ODEs, $u$ will be perfectly representable using a few parameters, e.g phase and frequency of an harmonic oscillator, and an analytical solution will be probably available.
If this is the case, one could rely on direct and efficient approaches,as we have show in [Example 01.00](examples/01.00-Logistic.jl), or a more general framework like the one described [here](https://docs.sciml.ai/DiffEqDocs/stable/examples/classical_physics/).

However most of the times, the analytical solution is not available. A common solution is the *method of lines*, that consists in discretizing the spacial dimension and study a vectorial representation of the function $u(x,t) \rightarrow \bar{u}(t)\in \mathbb{R}^N$, where $N\ll D$. This discretization can also be interpreted as a **filter** $\Phi\in\mathbb{R}^{D\times N}$ that gets applied to the field:
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

3. [Examples 02.03](examples/02.03-GrayScott.jl): DNS, LES and *exact* solutions.

The answer is that it depends on the harshness of the discretization, i.e. the value of $N$ for the specific problem.
We distinguish two cases:
* **DNS** (direct numerical simulation): if $N$ is large then eq.(2.2.2) is true, or we can say that the system is effectively *closed* because the discretized solution evolves following the behavior of the original system.

However, especially if the system is chaotic, it is likely that the small scale behavior influences the large scale behavior. In this case we talk about:
* **LES** (large eddy simulations): $N$ is too small to capture the small scale effects, thus the eq.(2.2.1) is *not closed*. This means that $\frac{d\bar{u}}{dt} = f_h(\bar{u})$ has a different dynamics than the exact solution $u(t)$, so we would need instead to solve  $$\frac{d\bar{u}}{dt} = \Phi f(u).$$ 
However this DE depends on $u$, so the system is not closed and can not be efficiently solved.

One of the main goal of this repository will be to use CNODEs like eq.(2.1.0) in order to correct the LES simulation and get a solution as good as the DNS.

4. [Examples 02.04](examples/02.04-GrayScott.jl): Trainable LES.

**... TO BE CONTINUED ...**



### Example 4 : Burgers
Look [here](https://github.com/DEEPDIP-project/NeuralNS-SciML-Tutorials). It has to be adapted, generalized and moved to this repo.

### Example 5 : Navier-Stokes
It is [here](https://github.com/DEEPDIP-project/NeuralNS-SciML-Tutorials). Make it compatible with this repo.

## References

* [SciML style guidelines](https://github.com/SciML/SciMLStyle)
