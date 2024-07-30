using CairoMakie
using IncompressibleNavierStokes
# [!] you can not use the abbreviation 'INS' with Enzyme, because it does not know how to differentiate the getname function!
INS = IncompressibleNavierStokes



# Setup and initial condition
T = Float32
ArrayType = Array
Re = T(1_000)
n = 256
n = 128
lims = T(0), T(1)
x , y = LinRange(lims..., n + 1), LinRange(lims..., n + 1)
setup = Setup(x, y; Re, ArrayType);
ustart = random_field(setup, T(0));
psolver = psolver_spectral(setup);
# [!] can not use spectral solver with Enzyme
# [https://github.com/JuliaMath/AbstractFFTs.jl/issues/99] we cannot define reverse-mode rules for mul!(y::AbstractArray, plan::Plan, x::AbstractArray), since we don't know what the plan is and therefore don't know how to normalize.
# The reason underneath is connected to Complex numbers as explained in Enzyme FAQ.
# one could do this to ignore the poisson solver
#EnzymeRules.inactive(::typeof(IncompressibleNavierStokes.poisson!), args...) = nothing
# but this is a cheat that breaks completely a posteriori fitting.
# Alternatively, you can use different poisson solvers
psolver = IncompressibleNavierStokes.psolver_direct(setup)
cache_p = INS.pressure(ustart, nothing, 0.0f0, setup; psolver)


dt = T(1e-3)
trange = [T(0), T(0.1)]
savevery = 20
saveat = 20 * dt


# Solving using INS semi-implicit method
(state, outputs), time_ins, allocation, gc, memory_counters = @timed solve_unsteady(;
    setup,
    ustart,
    tlims = trange,
    Δt = dt,
    psolver = psolver,
    processors = (
        ehist = realtimeplotter(;
            setup,
            plot = energy_history_plot,
            nupdate = 10,
            displayfig = false
        ),
        anim = animator(; setup, path = "./vorticity.mkv", nupdate = savevery),
        #espec = realtimeplotter(; setup, plot = energy_spectrum_plot, nupdate = 10),
        log = timelogger(; nupdate = 100)
    )
);


############# Using SciML
using DifferentialEquations

# Projected force for SciML, to use in CNODE
F = similar(stack(ustart))
# and prepare a cache for the force
cache_F = (F[:,:,1], F[:,:,2])
cache_div = INS.divergence(ustart,setup)
cache_p = INS.pressure(ustart, nothing, 0.0f0, setup; psolver)
cache_out = similar(F)
Ω = setup.grid.Ω

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
    for α = 1:D
        a, b = boundary_conditions[α]
        i = Ip[ntuple(β -> α == β ? (2:Np[α]-1) : (:), D)...][:]
        ia = Ip[ntuple(β -> α == β ? (1:1) : (:), D)...][:]
        ib = Ip[ntuple(β -> α == β ? (Np[α]:Np[α]) : (:), D)...][:]
        for (aa, bb, j) in [(a, nothing, ia), (nothing, nothing, i), (nothing, b, ib)]
            vala = @.(Ω[j] / Δ[α][getindex.(j, α)] / Δu[α][getindex.(j, α)-1])
            if isnothing(aa)
                J = [J; j .- [e(α)]; j]
                I = [I; j; j]
                val = [val; vala; -vala]
            elseif aa isa PressureBC
                J = [J; j]
                I = [I; j]
                val = [val; -vala]
            elseif aa isa PeriodicBC
                J = [J; ib; j]
                I = [I; j; j]
                val = [val; vala; -vala]
            elseif aa isa SymmetricBC
                J = [J; ia; j]
                I = [I; j; j]
                val = [val; vala; -vala]
            elseif aa isa DirichletBC
            end

            valb = @.(Ω[j] / Δ[α][getindex.(j, α)] / Δu[α][getindex.(j, α)])
            if isnothing(bb)
                J = [J; j; j .+ [e(α)]]
                I = [I; j; j]
                val = [val; -valb; valb]
            elseif bb isa PressureBC
                J = [J; j]
                I = [I; j]
                val = [val; -valb]
            elseif bb isa PeriodicBC
                J = [J; j; ia]
                I = [I; j; j]
                val = [val; -valb; valb]
            elseif bb isa SymmetricBC
                J = [J; j; ib]
                I = [I; j; j]
                val = [val; -valb; valb]
            elseif bb isa DirichletBC
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
    isdefinite =
        any(bc -> bc[1] isa PressureBC || bc[2] isa PressureBC, boundary_conditions)
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
# get the cache for the poisson solver
cache_ftemp, cache_ptemp, fact, cache_viewrange, cache_Ip = my_cache_psolver(setup.grid.x[1], setup)
# make fact into a global constant since it is a sparse array that it is complicated to express in Enzyme
global fact 
# Instead the other two can not be made global otherwise they break the stride
#global viewrange = cache_viewrange
#global Ip = cache_Ip

# This is the function that solves the Poisson equation
# notice that it is using the global constant fact
function my_psolve!(p, f, ftemp, ptemp, viewrange, Ip)
    copyto!(view(ftemp, viewrange), view(view(f, Ip), :))
    #ftemp[viewrange] .= vec(f[Ip])
    ptemp .= fact \ ftemp
    copyto!(view(view(p, Ip), :), eltype(p).(view(ptemp, viewrange)))
    nothing
end


# Requirements for Enzyme autodiff
# (1) you need to be able to do similar
# (2) SciML wants only some specific structure [https://docs.sciml.ai/SciMLStructures/stable/interface/]
# so it is not possible to solve (1) by defining a struct with its similar method like this
#struct MyCache
#    f::Tuple{Matrix{Float32}, Matrix{Float32}}
#    div::Matrix{Float32}
#    p::Matrix{Float32}
#    ftemp::Vector{Float32}
#    ptemp::Vector{Float32}
#    viewrange::UnitRange{Int64}
#    Ip::CartesianIndices{2, Tuple{UnitRange{Int64}, UnitRange{Int64}}}
#    θ::Any
#end
#function Base.similar(x::MyCache)
#    MyCache(
#        (similar(x.f[1]),similar(x.f[2])),
#        similar(x.div),
#        similar(x.p),
#        similar(x.ftemp),
#        similar(x.ptemp),
#        x.viewrange,
#        x.Ip,
#        x.θ
#    )
#end
# Instead, we can use ArrayPartition and overwrite the similar method in Base.jl 
using RecursiveArrayTools
function Base.similar(x::Tuple{Matrix{Float32}, Matrix{Float32}})
    (similar(x[1]),similar(x[2]))
end
function Base.similar(x::UnitRange{Int64})
    x
end
function Base.similar(x::CartesianIndices{2, Tuple{UnitRange{Int64}, UnitRange{Int64}}})
    x
end
# we also have to overwrite Base.zero
function Base.zero(x::Tuple{Matrix{Float32}, Matrix{Float32}})
    (zero(x[1]), zero(x[2]))
end
function Base.zero(x::UnitRange{Int64})
    x
end
function Base.zero(x::CartesianIndices{2, Tuple{UnitRange{Int64}, UnitRange{Int64}}})
    x
end
# Unfortunately it does not even work for ArrayPartition!
# let's try with a ComponentArray, for which we define the similar method
function Base.similar(x::Tuple{Tuple{Matrix{Float32}, Matrix{Float32}}, Matrix{Float32}, Matrix{Float32}, Vector{Float32}, Vector{Float32}, UnitRange{Int64}, CartesianIndices{2, Tuple{UnitRange{Int64}, UnitRange{Int64}}}, Float64})
    ComponentArray((similar(x[1]), similar(x[2]), similar(x[3]), similar(x[4]), similar(x[5]), x[6], x[7], x[8]))
end
function Base.zero(x::Tuple{Tuple{Matrix{Float32}, Matrix{Float32}}, Matrix{Float32}, Matrix{Float32}, Vector{Float32}, Vector{Float32}, UnitRange{Int64}, CartesianIndices{2, Tuple{UnitRange{Int64}, UnitRange{Int64}}}, Float64})
    ComponentArray((zero(x[1]), zero(x[2]), zero(x[3]), zero(x[4]), zero(x[5]), x[6], x[7], x[8]))
end
 



# **********************8
# * Force in place
F_ip(du, u, p, t)= begin
    u_view = eachslice(u; dims = 3)
    IncompressibleNavierStokes.apply_bc_u!(u_view, t, setup)
    IncompressibleNavierStokes.momentum!(p.x[1], u_view, nothing, t, setup)
    IncompressibleNavierStokes.apply_bc_u!(p.x[1], t, setup; dudt = true)
    #########
    IncompressibleNavierStokes.divergence!(p.x[2], p.x[1], setup)
    @. p.x[2] *= Ω
    # Solve the Poisson equation
    my_psolve!(p.x[3], p.x[2], p.x[4], p.x[5], p.x[6], p.x[7])
    IncompressibleNavierStokes.apply_bc_p!(p.x[3], T(0), setup)
    # Apply pressure correction term
    IncompressibleNavierStokes.applypressure!(p.x[1], p.x[3], setup)
    #########
    IncompressibleNavierStokes.apply_bc_u!(p.x[1], t, setup; dudt = true)
    du[:,:,1] .= p.x[1][1]
    du[:,:,2] .= p.x[1][2]
    nothing
end
temp = similar(stack(ustart))
# Define the cache for the force as an ArrayPartition
# this is needed to pass it to SciMLSensitivity and Enzyme  
# notice that the last place of the cache will be the p_nn ComponentArray
P = ArrayPartition(cache_F,cache_div, cache_p, cache_ftemp, cache_ptemp, cache_viewrange, cache_Ip, 0.0)
F_ip(temp, stack(ustart), P, 0.0f0);

P = ComponentArray((cache_F,cache_div, cache_p, cache_ftemp, cache_ptemp, cache_viewrange, cache_Ip, 0.0))
typeof(ComponentArray((cache_F,cache_div, cache_p, cache_ftemp, cache_ptemp, cache_viewrange, cache_Ip, 0.0)))
P[end]
o = similar(P)

# Solve the ODE using ODEProblem
prob = ODEProblem{true}(F_ip, stack(ustart), trange, p=P)
sol_ode, time_ode, allocation_ode, gc_ode, memory_counters_ode = @timed solve(
    prob,
    p = P,
    RK4();
    dt = dt,
    saveat = saveat,
);


import Random, Lux;
Random.seed!(123);
rng = Random.default_rng();


dummy_NN = Lux.Chain(
    Lux.Scale((1,1)),
)
θ_node, st_node = Lux.setup(rng, dummy_NN)
using ComponentArrays
θ_node = ComponentArray(θ_node)
# Test with a useless NN
θ_node.weight = [0.0f0;;]
θ_node.bias= [0.0f0;;]
Lux.apply(dummy_NN, stack(ustart), θ_node, st_node)[1];

# Force+NN in-place version
dudt_nn(du::Array{Float32}, u::Array{Float32}, P::ArrayPartition, t::Float32) = begin
    F_ip(du, u, P, t) 
    tmp = Lux.apply(dummy_NN, u, P.x[end], st_node)[1]
    @. du .= du .+ tmp
    nothing
end
#I can use ArrayPartition, but there is no similar  method
P = ArrayPartition(cache_F,cache_div, cache_p, cache_ftemp, cache_ptemp, cache_viewrange, cache_Ip, θ_node)

dudt_nn(temp, stack(ustart), P, 0.0f0)
prob_node = ODEProblem{true}(dudt_nn, stack(ustart), trange, p=P)

u0stacked = stack(ustart)
sol_node, time_node, allocation_node, gc_node, memory_counters_node = @timed solve(prob_node, RK4(), u0 = u0stacked, p = P, saveat = saveat, dt=dt);




# Compare the times of the different methods via a bar plot
using Plots
p1=Plots.bar(["INS", "ODE", "CNODE"], [time_ins, time_ode, time_node], xlabel = "Method", ylabel = "Time (s)", title = "Time comparison")
# Compare the memory allocation
p2=Plots.bar(["INS", "ODE", "CNODE"], [memory_counters.allocd, memory_counters_ode.allocd, memory_counters_node.allocd], xlabel = "Method", ylabel = "Memory (bytes)", title = "Memory comparison")
# Compare the number of garbage collections
p3=Plots.bar(["INS", "ODE", "CNODE"], [gc, gc_ode, gc_node], xlabel = "Method", ylabel = "Number of GC", title = "GC comparison")

Plots.plot(p1, p2, p3, layout=(3,1), size=(600, 800))



# Plot the final state
using Plots
p1=Plots.heatmap(title="u in SciML ODE",sol_ode.u[end][:, :, 1])
p2=Plots.heatmap(title="u in SciML CNODE",sol_node.u[end][:, :, 1])
p3=Plots.heatmap(title="u in INS",state.u[1])
# and compare them
p4=Plots.heatmap(title="u_INS-u_ODE",state.u[1] - sol_ode.u[end][:, :, 1])
p5=Plots.heatmap(title="u_INS-u_CNODE",state.u[1] - sol_node.u[end][:, :, 1])
p6=Plots.heatmap(title="u_CNODE-u_ODE",sol_node.u[end][:, :, 1] - sol_ode.u[end][:, :, 1])
Plots.plot(p1, p2, p3, p4,p5,p6, layout=(2,3), size=(900,600))


# Compute the divergence of the final state
div_INS = INS.divergence(state.u, setup);
div_ode = INS.divergence((sol_ode.u[end][:,:,1],sol_ode.u[end][:,:,2]), setup);
div_node = INS.divergence((sol_node.u[end][:,:,1],sol_node.u[end][:,:,2]), setup);
p1 = Plots.heatmap(title="div_INS",div_INS)
p2 = Plots.heatmap(title="div_ODE",div_ode)
p3 = Plots.heatmap(title="div_NODE",div_node)
Plots.plot(p1, p2, p3, layout=(1,3), size=(900,300))

########################
# Test the autodiff using Enzyme 
using Enzyme
using ComponentArrays
using SciMLSensitivity


# First test Enzyme for something that does not make sense bu it has the structure of a priori loss
U = stack(state.u) 
function fen(u0, p, temp, U)
    # Compute the force in-place
    dudt_nn(temp, u0, p, 0.0f0)
    return sum(U .- temp)
end
u0stacked = stack(ustart);
du = Enzyme.make_zero(u0stacked);
dU = Enzyme.make_zero(U);
dP = Enzyme.make_zero(P);
dtemp = Enzyme.make_zero(temp);
pino = similar(P)
# Compute the autodiff using Enzyme
Enzyme.autodiff(Enzyme.Reverse, fen, Active, Duplicated(u0stacked, du), Duplicated(P, dP), Duplicated(temp, dtemp), Duplicated(U, dU))
# the gradient that we need is only the following
dP.x[end]
# this shows us that Enzyme can differentiate our force. But what about SciML solvers?



# Define a posteriori loss function that calls the ODE solver
# First, make a shorter run
trange = [T(0), T(2*dt)]
saveat = dt
prob = ODEProblem{true}(F_ip, u0stacked, trange, p=P)
ode_data = Array(solve(prob, RK4(), u0 = u0stacked, p = P, saveat = saveat, dt=dt))
# the loss has to be in place 
function loss(l::Vector{Float32},θ::ArrayPartition, u0::Array{Float32}, tspan::Vector{Float32}, t::Float32)
    myprob = ODEProblem{true}(dudt_nn, u0, tspan)
    pred = Array(solve(myprob, RK4(), u0 = u0, p = θ, saveat=t, verbose=false))
    l .= Float32(sum(abs2, ode_data - pred))
    nothing
end
l=[0.0f0]
loss(l,P, u0stacked,trange, saveat)
l


# Test if the loss can be autodiffed
# [!] dl is called the 'seed' and it has to be marked to be one for correct gradient
l = [0.0f0]
dl = Enzyme.make_zero(l) .+1
dP = Enzyme.make_zero(P)
Enzyme.autodiff(Enzyme.Reverse, loss, Duplicated(l, dl), Duplicated(P, dP), Duplicated(u0stacked, du) , Const(trange), Const(saveat))
dP.x[end]
show(err)



##33 from here  [..]
##33 from here  [..]
##33 from here  [..]



function loss_gradient(G, θ::ComponentVector{Float32}, P1_u0_tspan_t_dP1_du0::Any) 
    P1, u0, tspan, t, dP1, du0 = P1_u0_tspan_t_dP1_du0
    # Reset gradient to zero
    Enzyme.make_zero!(G)
    # And remember to pass the seed to the loss funciton with the dual part set to 1
    Enzyme.autodiff(Enzyme.Reverse, lip, Duplicated([0.0f0], [1.0f0]), Duplicated(θ, G), DuplicatedNoNeed(u0, du0) , Const(tspan), Const(t))
    nothing
end

G = copy(dp)
oo = my_G(G,p_nn, [u0, tspan, t, du0])


# Then I can test Enzyme for a posteriori loss 
# pre-allcoate
pred = solve(prob_node, RK4(), u0 = u0stacked, p = (P1,pinit), saveat = saveat, sensealg=SciMLSensitivity.EnzymeVJP(), dt=dt)[1]
function myloss(θ::ComponentArray{Float32}, u0::Array{Float32}, P1)
    #pred = Array(solve(prob_node, RK4(), u0 = u0, p = (P1,θ), saveat = saveat, sensealg=SciMLSensitivity.EnzymeVJP(), dt=dt))
    pred = Array(solve(prob_node, RK4(), u0 = u0, p = (P1,θ), saveat = saveat, dt=dt))
    return sum(abs2, ode_data .- pred)
end
callback = function (θ, l; doplot = false)
    println(l)
    return false
end
callback(pinit, myloss(pinit, u0stacked, P1)...)
myloss(pinit, u0stacked, P1)

Enzyme.autodiff(Enzyme.Reverse, myloss, Active, Duplicated(pinit, dtheta), Duplicated(u0stacked, du), Duplicated(P1, dual_P1))
show(err)


using SciMLSensitivity
using Optimization, OptimizationOptimisers, Optimisers
# Select the autodifferentiation type
#adtype = Optimization.AutoZygote()
adtype = Optimization.AutoEnzyme()
# We transform the NeuralODE into an optimization problem
optf = Optimization.OptimizationFunction((x, p) -> myloss(x), adtype);
optprob = Optimization.OptimizationProblem(optf, pinit);
# And train using Adam + clipping
ClipAdam = OptimiserChain(Adam(1.0f-1), ClipGrad(1));
algo = ClipAdam
algo = OptimizationOptimisers.Adam(1.0f-1)
#using OptimizationCMAEvolutionStrategy, Statistics
#algo = CMAEvolutionStrategyOpt();
#import OptimizationOptimJL: Optim
#algo = Optim.LBFGS();
# ** train loop
result_neuralode = Optimization.solve(
    optprob,
    algo,
    callback = callback,
    maxiters = 10)
# You can continue the training from here
pinit = result_neuralode.u
θ = pinit;
optprob = Optimization.OptimizationProblem(optf, pinit);


# You have to write the gradient such that it can be read by Enzyme, because it seems like there is no space to store the gradient or something 