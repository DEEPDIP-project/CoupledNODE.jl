using CairoMakie
using IncompressibleNavierStokes
# [!] you can not use the abbreviation 'INS' with Enzyme, because it does not know how to differentiate the getname function!
INS = IncompressibleNavierStokes


#test componentarrays separating fx and fy 
#retest abstractarray



# Setup and initial condition
T = Float32
const myzero = T(0)
ArrayType = Array
Re = T(1_000)
n = 256
n = 128
n = 64
#n = 16
#n = 8
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
# I then compile those parameters into a psolver function
function generate_psolver(viewrange, Ip, fact)
    function psolver(p, f, ftemp, ptemp)
        copyto!(view(ftemp, viewrange), view(view(f, Ip), :))
        ptemp .= fact \ ftemp
        copyto!(view(view(p, Ip), :), eltype(p).(view(ptemp, viewrange)))
        nothing
    end
end
my_psolve! = generate_psolver(cache_viewrange, cache_Ip, fact)


##########################3
# There is a problem wiht the boundary conditions on p:
# INS can be differentiated a priori but not a posteriori
# my custom implementation down here can be differentiated only a posteriori 
# ??????
function myapply_bc_p!(p, t, setup; kwargs...)
    (; boundary_conditions, grid) = setup
    (; dimension) = grid
    D = dimension()
    for β = 1:D
        myapply_bc_p!(boundary_conditions[β][1], p, β, t, setup; isright = false)
        myapply_bc_p!(boundary_conditions[β][2], p, β, t, setup; isright = true)
    end
    p
end
function myapply_bc_p!(::PeriodicBC, p, β, t, setup; isright, kwargs...)
    (; grid, workgroupsize) = setup
    (; dimension, N) = grid
    D = dimension()
    e = Offset{D}()
    
    function _bc_a!(p, β)
        for I in CartesianIndices(p)
            I_β = I[β]
            if I_β == 1
                p[I] = p[I + (N[β] - 2) * e(β)]
            end
        end
    end

    function _bc_b!(p, β)
        for I in CartesianIndices(p)
            I_β = I[β]
            if I_β == N[β]
                p[I] = p[I - (N[β] - 2) * e(β)]
            end
        end
    end

    ndrange = ntuple(γ -> γ == β ? 1 : N[γ], D)
    
    if isright
        _bc_b!(p, β)
    else
        _bc_a!(p, β)
    end
    
    nothing
end


@timed for i in 1:1000
    A = rand(Float32,size(cache_p)[1],size(cache_p)[2]);
    IncompressibleNavierStokes.apply_bc_p!(A, 0.0f0, setup);
end
@timed for i in 1:1000
    A = rand(Float32,size(cache_p)[1],size(cache_p)[2]);
    myapply_bc_p!(A, 0.0f0, setup);
end

# Check if the implementation is correct
for i in 1:1000
    A = rand(Float32,size(cache_p)[1],size(cache_p)[2]) ;
    A0 = copy(A)                   ;
    B = copy(A)                    ;
    IncompressibleNavierStokes.apply_bc_p!(A, myzero, setup)  ;
    myapply_bc_p!(B, myzero, setup);
    @assert A ≈ B                  
end








# Requirements for Enzyme autodiff
# (1) you need to be able to do similar
# (2) SciML wants only some specific structure [https://docs.sciml.ai/SciMLStructures/stable/interface/]
# so it is not possible to solve (1) by defining a struct with its similar method like this
#struct MyCache
#    f::Tuple{Matrix{Float32}, Matrix{Float32}}
#    div::Matrix{Float32}
#    p::Matrix{Float32}
#    ft::Vector{Float32}
#    pt::Vector{Float32}
##    viewrange::UnitRange{Int64}
##    Ip::CartesianIndices{2, Tuple{UnitRange{Int64}, UnitRange{Int64}}}
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
# Instead, we can use ArrayPartition and overwrite the similar and zero methods in Base.jl 
## Unfortunately it does not even work for ArrayPartition!
# What is the problem with ArrayPartition?
# ArrayPartition produces the segmentation fault
# 
# Let's try with a ComponentArray!
# If we separate fx and fy in the cache it works!



# ------ Use Lux to create a dummy_NN
import Random, Lux;
Random.seed!(123);
rng = Random.default_rng();
dummy_NN = Lux.Chain(
    Lux.Scale((1,1)),
)
θ_node, st_node = Lux.setup(rng, dummy_NN)
using ComponentArrays
θ_node = ComponentArray(θ_node)
# You can set it to 0 like this
#θ_node.weight = [0.0f0;;]
#θ_node.bias= [0.0f0;;]
Lux.apply(dummy_NN, stack(ustart), θ_node, st_node)[1];


# Define the cache for the force 
using ComponentArrays
P = ComponentArray((f=F,div=cache_div, p=cache_p, ft=cache_ftemp, pt=cache_ptemp, θ=θ_node, a_priori=1))
# And gets its type
TP = typeof(P)


# **********************8
# * Force in place
F_ip(du::Array{Float32}, u::Array{Float32}, p::TP, t::Float32) = begin
    u_view = eachslice(u; dims = 3)
    F = eachslice(p.f; dims = 3)
    IncompressibleNavierStokes.apply_bc_u!(u_view, t, setup)
    IncompressibleNavierStokes.momentum!(F, u_view, nothing, t, setup)
    IncompressibleNavierStokes.apply_bc_u!(F, t, setup; dudt = true)
    #########
    IncompressibleNavierStokes.divergence!(p.div, F, setup)
    @. p.div *= Ω
    # Solve the Poisson equation
    my_psolve!(p.p, p.div, p.ft, p.pt)
    # There are some problems with the boundary conditions on p so I had to redefine the function for a priori differentiation
    if p.a_priori == 1
        myapply_bc_p!(p.p, myzero, setup)
    else
        IncompressibleNavierStokes.apply_bc_p!(p.p, myzero, setup)
    end
    # Apply pressure correction term
    IncompressibleNavierStokes.applypressure!(F, p.p, setup)
    #########
    IncompressibleNavierStokes.apply_bc_u!(F, t, setup; dudt = true)
    du[:,:,1] .= F[1]
    du[:,:,2] .= F[2]
    nothing
end
temp = similar(stack(ustart))
F_ip(temp, stack(ustart), P, 0.0f0)


# Solve the ODE using ODEProblem
prob = ODEProblem{true}(F_ip, stack(ustart), trange, p=P)
sol_ode, time_ode, allocation_ode, gc_ode, memory_counters_ode = @timed solve(
    prob,
    p = P,
    RK4();
    dt = dt,
    saveat = saveat,
);



# Force+NN in-place version
dudt_nn(du::Array{Float32}, u::Array{Float32}, P::TP, t::Float32) = begin
    F_ip(du, u, P, t) 
    #tmp = Lux.apply(dummy_NN, u, P.x[end], st_node)[1]
    tmp = Lux.apply(dummy_NN, u, P.θ , st_node)[1]
    @. du .= du .+ tmp
    nothing
end


dudt_nn(temp, stack(ustart), P, 0.0f0)
prob_node = ODEProblem{true}(dudt_nn, stack(ustart), trange, p=P)

u0stacked = stack(ustart)
sol_node, time_node, allocation_node, gc_node, memory_counters_node = @timed solve(prob_node, RK4(), u0 = u0stacked, p = P, saveat = saveat, dt=dt);


println("Done run")

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
function fen(u0::Array{Float32}, p::TP, temp::Array{Float32})
    # Compute the force in-place
    dudt_nn(temp, u0, p, 0.0f0)
    return sum(U .- temp)
end
u0stacked = stack(ustart);
du = Enzyme.make_zero(u0stacked);
dP = Enzyme.make_zero(P);
temp = similar(stack(ustart))
dtemp = Enzyme.make_zero(temp);
# Compute the autodiff using Enzyme
@timed Enzyme.autodiff(Enzyme.Reverse, fen, Active, DuplicatedNoNeed(u0stacked, du), DuplicatedNoNeed(P, dP), DuplicatedNoNeed(temp, dtemp))
# the gradient that we need is only the following
dP.θ
# this shows us that Enzyme can differentiate our force. But what about SciML solvers?
println("Tested a priori")
show(err)


# Define a posteriori loss function that calls the ODE solver
# First, make a shorter run
trange = [T(0), T(10*dt)];
saveat = dt;
prob = ODEProblem{true}(F_ip, u0stacked, trange, p=P);
ode_data = Array(solve(prob, RK4(), u0 = u0stacked, p = P, saveat = saveat, dt=dt));
# the loss has to be in place 
function loss(l::Vector{Float32},θ::TP, u0::Array{Float32}, tspan::Vector{Float32}, t::Float32)
    myprob = ODEProblem{true}(dudt_nn, u0, tspan)
    pred = Array(solve(myprob, RK4(), u0 = u0, p = θ, saveat=t, verbose=false))
    l .= Float32(sum(abs2, ode_data - pred))
    nothing
end
l=[0.0f0];
loss(l,P, u0stacked,trange, saveat);
l



# Test if the loss can be autodiffed
# [!] dl is called the 'seed' and it has to be marked to be one for correct gradient
l = [0.0f0];
dl = Enzyme.make_zero(l) .+1;
P = ComponentArray((f=F,div=cache_div, p=cache_p, ft=cache_ftemp, pt=cache_ptemp, θ=θ_node, a_priori=0));
dP = Enzyme.make_zero(P);
du = Enzyme.make_zero(u0stacked);
@timed Enzyme.autodiff(Enzyme.Reverse, loss, DuplicatedNoNeed(l, dl), DuplicatedNoNeed(P, dP), DuplicatedNoNeed(u0stacked, du) , Const(trange), Const(saveat))
dP.θ
    
    
    
    
extra_par = [u0stacked, trange, saveat, du, dP, P];
Textra = typeof(extra_par);
Tth = typeof(P.θ);
#function loss_gradient(G::Tth, θ::Tth, extra_par::Textra) 
function loss_gradient(G, extra_par) 
    u0, trange, saveat, du0, dP, P = extra_par
    # [!] Notice that we are updating P.θ in-place in the loss function
    # Reset gradient to zero
    Enzyme.make_zero!(dP)
    # And remember to pass the seed to the loss funciton with the dual part set to 1
    Enzyme.autodiff(Enzyme.Reverse, loss, DuplicatedNoNeed([0.0f0], [1.0f0]), DuplicatedNoNeed(P,dP), DuplicatedNoNeed(u0, du0) , Const(trange), Const(saveat))
    # The gradient matters only for theta
    G .= dP.θ
    nothing
end

G = copy(dP.θ);
oo = loss_gradient(G, extra_par)


# This is to call loss using only P
#function over_loss(θ::Tth, p::TP)
function over_loss(θ, p)
    # Here we are updating P.θ in place
    p.θ .= θ
    loss(l,p, u0stacked,trange, saveat)
    return l
end
callback = function (θ,l; doplot = false)
    println(l)
    return false
end
callback(P, over_loss(P.θ, P))


using SciMLSensitivity, Optimization, OptimizationOptimisers, Optimisers
optf = Optimization.OptimizationFunction((p,u)->over_loss(p,u[end]), grad=(G,p,e)->loss_gradient(G,e))
#optf = Optimization.OptimizationFunction((p,u)->over_loss(p,u[end]), grad=(G,p,e)->println("\n------\nG: ",typeof(G),"\np: ", typeof(p)) )
optprob = Optimization.OptimizationProblem(optf, P.θ, extra_par)


result_e, time_e, alloc_e, gc_e, mem_e = @timed Optimization.solve(optprob,
    OptimizationOptimisers.Adam(0.05),
    callback = callback,
    maxiters = 100)


# Zygote can not be used because we are mutating, then how do we compare?