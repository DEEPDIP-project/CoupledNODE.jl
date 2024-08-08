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
u0 = stack(ustart)
psolver = INS.psolver_spectral(setup);
dt = T(1e-3)
trange = (T(0), T(10))
savevery = 20
saveat = savevery * dt;

# ## DNS data

state, outputs = INS.solve_unsteady(; setup, ustart, tlims = trange, Δt = dt,
    processors = (field = INS.fieldsaver(; setup, nupdate = savevery),
        log = INS.timelogger(; nupdate = 100))
);

#one can get the entire u, see here how to acces a specific time step:
#outputs.field[step].u[1]

# ## Projected force for SciML

using DifferentialEquations
F = similar(u0)
# and prepare a cache for the force
cache_F = (F[:, :, 1], F[:, :, 2])
cache_div = INS.divergence(ustart, setup)
cache_p = INS.pressure(ustart, nothing, 0.0f0, setup; psolver)
cache_out = similar(F)
Ω = setup.grid.Ω

# * Force in place
F_ip!(du, u, p, t) = begin
    u_view = eachslice(u; dims = 3)
    INS.apply_bc_u!(u_view, t, setup)
    INS.momentum!(p[1], u_view, nothing, t, setup)
    INS.apply_bc_u!(p[1], t, setup; dudt = true)
    INS.project!(p[1], setup; psolver = p[4], div = p[2], p = p[3])
    INS.apply_bc_u!(p[1], t, setup; dudt = true)
    du[:, :, 1] .= p[1][1]
    du[:, :, 2] .= p[1][2]
    nothing
end

temp_du = similar(u0)
myfull_cache = (cache_F, cache_div, cache_p, psolver)
F_ip!(temp_du, u0, myfull_cache, 0.0f0);

# We need to modify the solver in place in order to pass everything to Enzyme
using SparseArrays, LinearAlgebra
struct Offset{D} end
@inline (::Offset{D})(α) where {D} = CartesianIndex(ntuple(β -> β == α ? 1 : 0, D))
function laplacian_mat(setup)
    (; grid, boundary_conditions) = setup
    (; dimension, x, N, Np, Ip, Δ, Δu, Ω) = grid
    T = eltype(x[1])
    D = dimension()
    e = Offset{D}()
    Ia = first(Ip)
    Ib = last(Ip)
    I = similar(x[1], CartesianIndex{D}, 0)
    J = similar(x[1], CartesianIndex{D}, 0)
    val = similar(x[1], 0)
    I0 = Ia - oneunit(Ia)
    for α in 1:D
        a, b = boundary_conditions[α]
        i = Ip[ntuple(β -> α == β ? (2:(Np[α] - 1)) : (:), D)...][:]
        ia = Ip[ntuple(β -> α == β ? (1:1) : (:), D)...][:]
        ib = Ip[ntuple(β -> α == β ? (Np[α]:Np[α]) : (:), D)...][:]
        for (aa, bb, j) in [(a, nothing, ia), (nothing, nothing, i), (nothing, b, ib)]
            vala = @.(Ω[j] / Δ[α][getindex.(j, α)]/Δu[α][getindex.(j, α) - 1])
            if isnothing(aa)
                J = [J; j .- [e(α)]; j]
                I = [I; j; j]
                val = [val; vala; -vala]
            elseif aa isa INS.PressureBC
                J = [J; j]
                I = [I; j]
                val = [val; -vala]
            elseif aa isa INS.PeriodicBC
                J = [J; ib; j]
                I = [I; j; j]
                val = [val; vala; -vala]
            elseif aa isa INS.SymmetricBC
                J = [J; ia; j]
                I = [I; j; j]
                val = [val; vala; -vala]
            elseif aa isa INS.DirichletBC
            end

            valb = @.(Ω[j] / Δ[α][getindex.(j, α)]/Δu[α][getindex.(j, α)])
            if isnothing(bb)
                J = [J; j; j .+ [e(α)]]
                I = [I; j; j]
                val = [val; -valb; valb]
            elseif bb isa INS.PressureBC
                J = [J; j]
                I = [I; j]
                val = [val; -valb]
            elseif bb isa INS.PeriodicBC
                J = [J; j; ia]
                I = [I; j; j]
                val = [val; -valb; valb]
            elseif bb isa INS.SymmetricBC
                J = [J; j; ib]
                I = [I; j; j]
                val = [val; -valb; valb]
            elseif bb isa INS.DirichletBC
            end
        end
    end
    I = Array(I)
    J = Array(J)
    I = I .- [I0]
    J = J .- [I0]
    linear = LinearIndices(Ip)
    I = linear[I]
    J = linear[J]

    L = sparse(I, J, Array(val))
    L
end
function my_cache_psolver(::Array, setup)
    (; grid, boundary_conditions) = setup
    (; x, Np, Ip) = grid
    T = eltype(x[1])
    L = laplacian_mat(setup)
    isdefinite = any(
        bc -> bc[1] isa INS.PressureBC || bc[2] isa INS.PressureBC, boundary_conditions)
    if isdefinite
        println("Definite")
        # No extra DOF
        T = Float64 # This is currently required for SuiteSparse LU
        ftemp = zeros(T, prod(Np))
        ptemp = zeros(T, prod(Np))
        viewrange = (:)
        fact = factorize(L)
    else
        println("Indefinite")
        # With extra DOF
        ftemp = zeros(T, prod(Np) + 1)
        ptemp = zeros(T, prod(Np) + 1)
        e = ones(T, size(L, 2))
        L = [L e; e' 0]
        maximum(L - L') < sqrt(eps(T)) || error("Matrix not symmetric")
        L = @. (L + L') / 2
        viewrange = 1:prod(Np)
        fact = ldlt(L)
    end
    return ftemp, ptemp, fact, viewrange, Ip
end
cache_ftemp, cache_ptemp, fact, cache_viewrange, cache_Ip = my_cache_psolver(
    setup.grid.x[1], setup)

global fact
function my_psolve!(p, f, ftemp, ptemp, viewrange, Ip)
    copyto!(view(ftemp, viewrange), view(view(f, Ip), :))
    #ftemp[viewrange] .= vec(f[Ip])
    ptemp .= fact \ ftemp
    copyto!(view(view(p, Ip), :), eltype(p).(view(ptemp, viewrange)))
    nothing
end

# ## Zygote + Lux

import Random, Lux;
Random.seed!(123);
rng = Random.default_rng();
dummy_NN = Lux.Chain(Lux.Scale((1, 1)),)
zl_NN = Lux.Chain(
    Lux.Dense(2, 50, tanh),
    Lux.Dense(50, 2, identity)
)
θ_zl, st_zl = Lux.setup(rng, dummy_NN)
using ComponentArrays
θ_zl = ComponentArray(θ_zl)
# Test with a useless NN
θ_zl.weight = [0.0f0;;]
θ_zl.bias = [0.0f0;;]
Lux.apply(dummy_NN, u0, θ_zl, st_zl)[1];

dudt_nn_zl!(du, u, p, t) = begin
    F_ip!(du, u, p, t)
    tmp = Lux.apply(dummy_NN, u, p[end], st_zl)[1]
    @. du .= du .+ tmp
    nothing
end

zl_cache = (myfull_cache..., θ_zl)
dudt_nn_zl!(temp_du, u0, zl_cache, 0.0f0)
prob_zl = ODEProblem{true}(dudt_nn_zl!, u0, trange, p = myfull_cache)
sol_zl, time_zl, allocation_zl, gc_zl, memory_counters_zl = @timed solve(
    prob_zl, RK4(), u0 = u0, p = zl_cache, saveat = saveat, dt = dt);

# ## Zygote + SimpleChain
import SimpleChains as SC
sc_NN = SC.SimpleChain(SC.static(2), #TODO: what should be the input size?
    SC.TurboDense{true}(SC.tanh, SC.static(50)),
    SC.TurboDense{true}(identity, SC.static(2)))
θ_sc = Array(SC.init_params(sc_NN))
G_sc = SC.alloc_threaded_grad(sc_NN);

dudt_nn_sc!(du, u, p, t) = begin
    F_ip!(du, u, p, t)
    tmp = sc_NN(u, p[end])
    @. du .= du .+ tmp
    nothing
end

sc_cache = (myfull_cache..., θ_sc)
dudt_nn_sc!(temp_du, u0, sc_cache, 0.0f0)

# ## Enzyme + Lux

# ## Enzime + SimpleChain 
