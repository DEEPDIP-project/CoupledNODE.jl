![CNODE banner](https://raw.githubusercontent.com/DEEPDIP-project/CoupledNODE.jl/main/assets/logo_black.png#gh-light-mode-only)
![CNODE banner](https://raw.githubusercontent.com/DEEPDIP-project/CoupledNODE.jl/main/assets/logo_white.png#gh-dark-mode-only)

# CoupledNODE

[![Build Status](https://github.com/DEEPDIP-project/CoupledNODE.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/DEEPDIP-project/CoupledNODE.jl/actions/workflows/CI.yml)
[![Documentation Status](https://readthedocs.org/projects/gemdat/badge/?version=latest)](https://deepdip-project.github.io/CoupledNODE.jl/stable/)

## Installation

```julia
] # open Package manager
activate .
instantiate
```

## Neural ODEs

Let's consider a generic differential equation (DE):

$$
\begin{cases} \frac{du}{dt} = f(t,u) \\
u(t=0) = u_0 \end{cases} \tag{1.0}
$$

where we look for the time-dependent solution $u(t)$, given its initial condition $u_0$ and the force $f(t,u)$ acting on it.
In general, we have an ordinary differential equation (ODE) when $$\frac{du}{dt} = f(u, t)$$ contains a single independent variable $u$.

We have a Neural ODE (NODE) when the structure of the problem becomes: $$\frac{du}{dt} = f(u,t) + NN(u |\theta).$$ Notice the extra force term, named $NN$ because it can be a neural network, that depends on extra parameters $\theta$ that are not part of the solution. Overall, a NODE problem is a data-driven method that aims to model the solution $u(t)$ when the ODE can not be solved exactly, or even if $f$ is unknown. Let's look at the details with an example.

### Multiscale approach to chaotic PDEs
In a more realistic situation, the incognita is usually a vector field, so we will try to solve for $u(x,t): \Omega \times \mathbb{R} \rightarrow \mathbb{R}^m$, where the spacial coordinate is $x\in \Omega \subseteq \mathbb{R}^D$ and the field $u$ is $m$-dimensional.

If the problem is simple, like the case of linear ODEs, $u$ will be perfectly representable using a few parameters, e.g phase and frequency of an harmonic oscillator, and an analytical solution will be probably available.
If this is the case, one could rely on direct and efficient approaches.

However most of the times, the analytical solution is not available. A common solution is the *method of lines*, that consists in discretizing the spacial dimension and study a vectorial representation of the function $u(x,t) \rightarrow \bar{u}(t)\in \mathbb{R}^N$, where $N\ll D$. This discretization can also be interpreted as a **filter** $\Phi\in\mathbb{R}^{D\times N}$ that gets applied to the field:
$$\bar{u}= \Phi u$$

Once we have a discretized space, we also discretize the force $f\rightarrow f_h$ using finite differences, while also assuming that the force is not time dependent. So we are left with

$$
\tag{1}
\begin{cases}
\frac{d\bar{u}}{dt} = f_h(\bar{u}) \\
\bar{u}(t=0) = \Phi u(t=0) = \bar{u}_0
\end{cases}
$$

Then the crucial question that we need to address is

$$
\tag{2}\bar{u}(t)\stackrel{?}{=} \Phi u(t),
$$

or in words: **does the discretized solution evolve like the original?**

The answer is that it depends on the harshness of the discretization, i.e. the value of $N$ for the specific problem.
We distinguish two cases:
* **DNS** (direct numerical simulation): if $N$ is large then eq.(2) is true, or we can say that the system is effectively *closed* because the discretized solution evolves following the behavior of the original system.

However, especially if the system is chaotic, it is likely that the small scale behavior influences the large scale behavior. In this case we talk about:
* **LES** (large eddy simulations): $N$ is too small to capture the small scale effects, thus the eq.(1) is *not closed*. This means that $\frac{d\bar{u}}{dt} = f_h(\bar{u})$ has a different dynamics than the exact solution $u(t)$, so we would need instead to solve $$\frac{d\bar{u}}{dt} = \Phi f(u).$$

However this DE depends on $u$, so the system is not closed and can not be efficiently solved.

One of the main goal of this repository will be to use CNODEs in order to correct the LES simulation and get a solution as good as the DNS.

## References

* [SciML style guidelines](https://github.com/SciML/SciMLStyle)
