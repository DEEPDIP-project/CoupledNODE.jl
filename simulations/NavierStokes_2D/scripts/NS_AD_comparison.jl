# # Comparison of AD libraries for Navier-Stokes 2D + closure
# In this notebook we compare the *implementation and performance* of:
# 1. **AD** (Automatic differentiation) techniques: Zygote.jl vs Enzyme.jl
# 2. **ML frameworks**: Lux.jl vs SimpleChain.jl
# We will focus on solving Navier Stokes 2D (LES) with a closure model via apriori and a posteriori fitting.

# Setup and initial condition
using GLMakie
import IncompressibleNavierStokes as INS
T = Float32
ArrayType = Array
Re = T(1_000)
n = 64
lims = T(0), T(1)
x, y = LinRange(lims..., n + 1), LinRange(lims..., n + 1)
setup = INS.Setup(x, y; Re, ArrayType);
ustart = INS.random_field(setup, T(0));
psolver = INS.psolver_spectral(setup);
dt = T(1e-3)
trange = [T(0), T(10)]
savevery = 20
saveat = savevery * dt;

# Projected force for SciML