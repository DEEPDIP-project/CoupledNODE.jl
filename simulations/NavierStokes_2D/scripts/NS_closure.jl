using CairoMakie
using IncompressibleNavierStokes
# [!] you can not use the abbreviation 'INS' with Enzyme, because it does not know how to differentiate the getname function!
INS = IncompressibleNavierStokes


# [!] Do NOT use typing with Enzyme!

le mypbc non gli piaccion a enzyme perche mischiano l activity allora levale
puoi provare a imparare div=0 usando come target le simulazioni con INS

# Setup and initial condition
T = Float32
const myzero = T(0)
ArrayType = Array
Re = T(1_000)
n = 256
n = 128
n = 64
#n = 32
#n = 16
#n = 8
lims = T(0), T(1)
x , y = LinRange(lims..., n + 1), LinRange(lims..., n + 1)
const setup = Setup(x, y; Re, ArrayType);
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


##########################
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



############# Test a similar thing for the BC on u
using KernelAbstractions
function myapply_bc_u!(u, t, setup; kwargs...)
    (; boundary_conditions) = setup
    D = length(u)
    for β = 1:D
        myapply_bc_u!(boundary_conditions[β][1], u, β, t, setup; isright = false, kwargs...)
        myapply_bc_u!(boundary_conditions[β][2], u, β, t, setup; isright = true, kwargs...)
    end
    u
end
function myapply_bc_u!(::PeriodicBC, u, β, t, setup; isright, kwargs...)
    (; grid, workgroupsize) = setup
    (; dimension, N) = grid
    D = dimension()
    e = Offset{D}()
    
    function _bc_a!(u, α, β)
        for I in CartesianIndices(N)
            if I[β] == 1
                u[α][I] = u[α][I + (N[β] - 2) * e(β)]
            end
        end
    end
    
    function _bc_b!(u, α, β)
        for I in CartesianIndices(N)
            if I[β] == N[β]
                u[α][I] = u[α][I - (N[β] - 2) * e(β)]
            end
        end
    end
    
    for α = 1:D
        if isright
            _bc_b!(u, α, β)
        else
            _bc_a!(u, α, β)
        end
    end
    u
end


@timed for i in 1:1000
    A = (rand(Float32,size(cache_p)[1],size(cache_p)[1]),rand(Float32,size(cache_p)[1],size(cache_p)[1]))
    IncompressibleNavierStokes.apply_bc_u!(A, 0.0f0, setup);
end
@timed for i in 1:1000
    A = (rand(Float32,size(cache_p)[1],size(cache_p)[1]),rand(Float32,size(cache_p)[1],size(cache_p)[1]))
    myapply_bc_u!(A, 0.0f0, setup);
end

# Check if the implementation is correct
for i in 1:1000
    A = (rand(Float32,size(cache_p)[1],size(cache_p)[1]),rand(Float32,size(cache_p)[1],size(cache_p)[1]));
    A0 = (copy(A[1]), copy(A[2])) ;                  ;
    B = (copy(A[1]), copy(A[2]))                    ;
    IncompressibleNavierStokes.apply_bc_u!(A, myzero, setup)  ;
    myapply_bc_u!(B, myzero, setup);
    @assert A[1] ≈ B[1]                  
    @assert A[2] ≈ B[2]
end



###########################################
# **** Test if you can make divergence faster
function mydivergence!(div, u, setup)
    (; grid, workgroupsize) = setup
    (; Δ, N, Ip, Np) = grid
    D = length(u)
    e = Offset{D}()

    I0 = first(Ip)
    I0 -= oneunit(I0)

    @inbounds for idx in CartesianIndices(Np)
        I = idx + I0
        d = 0.0
        for α in 1:D
            uα = u[α]
            Δα = Δ[α]
            Iα = I[α]
            d += (uα[I] - uα[I - e(α)]) / Δα[Iα]
        end
        div[I] = d
    end

    div
end
@timed for i in 1:1000
    A = rand(Float32,size(cache_p)[1],size(cache_p)[2]);
    u = random_field(setup, T(0));
    IncompressibleNavierStokes.divergence!(A, u, setup);
end
@timed for i in 1:1000
    A = rand(Float32,size(cache_p)[1],size(cache_p)[2]);
    u = random_field(setup, T(0));
    mydivergence!(A, u, setup);
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
    Lux.ReshapeLayer(((n+2)*(n+2),)),
    Lux.Dense((n+2)*(n+2)=>(n+2)*(n+2),init_weight = Lux.WeightInitializers.zeros32),
    Lux.ReshapeLayer(((n+2),(n+2))),
)
# Scale can not be differentiated by Enzyme!
#dummy_NN = Lux.Chain(
#    Lux.Scale((1,1)),
#)
θ_node, st_node = Lux.setup(rng, dummy_NN)

using ComponentArrays
θ_node = ComponentArray(θ_node)
# You can set it to 0 like this
#θ_node.weight = [0.0f0;;]
#θ_node.bias= [0.0f0;;]
Lux.apply(dummy_NN, stack(ustart), θ_node, st_node)[1]

# Define the cache for the force 
using ComponentArrays
P = ComponentArray((f=zeros(T, (n+2,n+2,2)),div=zeros(T,(n+2,n+2)), p=zeros(T,(n+2,n+2)), ft=zeros(T,size(cache_ftemp)), pt=zeros(T,size(cache_ptemp)), lux=zeros(T,(n+2,n+2,2)), θ=copy(θ_node), a_priori=1))
# And gets its type
TP = typeof(P)


# **********************8
# * Force in place
#function F_ip(du::Array{Float32}, u::Array{Float32}, p::TP, t::Float32)
function F_ip(du, u, p, t)
    #u_view = eachslice(u; dims = 3)
    #F = eachslice(p.f; dims = 3)
    #if p.a_priori == 1
    #    myapply_bc_u!(u_view, t, setup)
    #else
    #    IncompressibleNavierStokes.apply_bc_u!(u_view, t, setup)
    #end
    #IncompressibleNavierStokes.momentum!(F, u_view, nothing, t, setup)
    #if p.a_priori == 1
    #    myapply_bc_u!(F, t, setup)
    #else
    #    IncompressibleNavierStokes.apply_bc_u!(F, t, setup)
    #end
    ##IncompressibleNavierStokes.divergence!(p.div, F, setup)
    ##@. p.div *= Ω
    ##my_psolve!(p.p, p.div, p.ft, p.pt)
    #if p.a_priori == 1
    #    myapply_bc_p!(p.p, myzero, setup)
    #else
    #    IncompressibleNavierStokes.apply_bc_p!(p.p, myzero, setup)
    #end
    #IncompressibleNavierStokes.applypressure!(F, p.p, setup)
    #if p.a_priori == 1
    #    myapply_bc_u!(F, t, setup)
    #else
    #    IncompressibleNavierStokes.apply_bc_u!(F, t, setup)
    #end
    #du[:,:,1] .= F[1]
    #du[:,:,2] .= F[2]
    #nothing
    u_view = eachslice(u; dims = 3)
    F = eachslice(p.f; dims = 3)
    myapply_bc_u!(u_view, t, setup)
    IncompressibleNavierStokes.momentum!(F, u_view, nothing, t, setup)
    myapply_bc_u!(F, t, setup)
    IncompressibleNavierStokes.divergence!(p.div, F, setup)
    @. p.div *= Ω
    my_psolve!(p.p, p.div, p.ft, p.pt)
    myapply_bc_p!(p.p, myzero, setup)
    IncompressibleNavierStokes.applypressure!(F, p.p, setup)
    myapply_bc_u!(F, t, setup)
    du[:,:,1] .= F[1]
    du[:,:,2] .= F[2]
    nothing
    #times = Dict{String, Float64}()

    #u_view = eachslice(u; dims = 3)
    #F = eachslice(p.f; dims = 3)

    #elapsed_time = @elapsed IncompressibleNavierStokes.apply_bc_u!(u_view, t, setup)
    #println("Time for apply_bc_u! (initial): ", elapsed_time, " seconds")
    #times["apply_bc_u! (initial)"] = elapsed_time

    #elapsed_time = @elapsed IncompressibleNavierStokes.momentum!(F, u_view, nothing, t, setup)
    #println("Time for momentum!: ", elapsed_time, " seconds")
    #times["momentum!"] = elapsed_time

    #elapsed_time = @elapsed IncompressibleNavierStokes.apply_bc_u!(F, t, setup; dudt = true)
    #println("Time for apply_bc_u! (dudt): ", elapsed_time, " seconds")
    #times["apply_bc_u! (dudt)"] = elapsed_time

    #elapsed_time = @elapsed IncompressibleNavierStokes.divergence!(p.div, F, setup)
    #println("Time for divergence!: ", elapsed_time, " seconds")
    #times["divergence!"] = elapsed_time

    #elapsed_time = @elapsed @. p.div *= Ω
    #println("Time for scaling p.div: ", elapsed_time, " seconds")
    #times["scaling p.div"] = elapsed_time

    #elapsed_time = @elapsed my_psolve!(p.p, p.div, p.ft, p.pt)
    #println("Time for my_psolve!: ", elapsed_time, " seconds")
    #times["my_psolve!"] = elapsed_time

    #if p.a_priori == 1
    #    elapsed_time = @elapsed myapply_bc_p!(p.p, myzero, setup)
    #    println("Time for myapply_bc_p!: ", elapsed_time, " seconds")
    #    times["myapply_bc_p!"] = elapsed_time
    #else
    #    elapsed_time = @elapsed IncompressibleNavierStokes.apply_bc_p!(p.p, myzero, setup)
    #    println("Time for apply_bc_p!: ", elapsed_time, " seconds")
    #    times["apply_bc_p!"] = elapsed_time
    #end

    #elapsed_time = @elapsed IncompressibleNavierStokes.applypressure!(F, p.p, setup)
    #println("Time for applypressure!: ", elapsed_time, " seconds")
    #times["applypressure!"] = elapsed_time

    #elapsed_time = @elapsed IncompressibleNavierStokes.apply_bc_u!(F, t, setup; dudt = true)
    #println("Time for apply_bc_u! (final): ", elapsed_time, " seconds")
    #times["apply_bc_u! (final)"] = elapsed_time

    #elapsed_time = @elapsed begin
    #    du[:,:,1] .= F[1]
    #    du[:,:,2] .= F[2]
    #end
    #println("Time for updating du: ", elapsed_time, " seconds")
    #times["updating du"] = elapsed_time

    ## Rank and print the times
    #sorted_times = sort(collect(times), by = x -> x[2], rev = true)
    #println("\nRanked execution times:")
    #for (operation, time) in sorted_times
    #    println("$operation: $time seconds")
    #end

    #nothing
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
#dudt_nn(du::Array{Float32}, u::Array{Float32}, P::TP, t::Float32) = begin
dudt_nn(du, u, P, t) = begin
    F_ip(du, u, P, t) 
    P.lux .= Lux.apply(dummy_NN, u, P.θ , st_node)[1]
    #@. du += P.lux
    du .= du .+ P.lux
    #du .= Lux.apply(dummy_NN, u, P.θ , st_node)[1]
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
#U = stack(state.u);
#function fen(u0::Array{Float32}, p::TP, temp::Array{Float32})
#    # Compute the force in-place
#    dudt_nn(temp, u0, p, 0.0f0)
#    return sum(U .- temp)
#end
u0stacked = stack(ustart);
du = Enzyme.make_zero(u0stacked);
P = ComponentArray((f=zeros(T, (n+2,n+2,2)),div=zeros(T,(n+2,n+2)), p=zeros(T,(n+2,n+2)), ft=zeros(T,size(cache_ftemp)), pt=zeros(T,size(cache_ptemp)), lux=zeros(T,(n+2,n+2,2)), θ=copy(θ_node), a_priori=0))
dP = Enzyme.make_zero(P);
temp = similar(stack(ustart))
dtemp = Enzyme.make_zero(temp);
# Compute the autodiff using Enzyme
#@timed Enzyme.autodiff(Enzyme.Reverse, fen, Active, DuplicatedNoNeed(u0stacked, du), DuplicatedNoNeed(P, dP), DuplicatedNoNeed(temp, dtemp))
# the gradient that we need is only the following
dP.θ
# this shows us that Enzyme can differentiate our force. But what about SciML solvers?
println("Tested a priori")
show(err)



# Define a posteriori loss function that calls the ODE solver
# First, make a shorter run
# and remember to set a small dt
dt = T(1e-4)
typeof(dt)
trange = [T(0), T(3*dt)];
saveat = dt;
prob = ODEProblem{true}(F_ip, u0stacked, trange, p=P);
ode_data = Array(solve(prob, RK4(), u0 = u0stacked, p = P, saveat = saveat, dt=dt));
ode_data += T(0.1)*rand(Float32, size(ode_data))

# the loss has to be in place 
#function loss(l::Vector{Float64},θ::TP, u0::Array{Float32}, tspan::Vector{Float32}, t::Float32, dt::Float32, pred::Array{Float32})
#function loss(l::Vector{Float32},P::TP, u0::Array{Float32}, pred::Array{Float32})
#function loss(l::Vector{Float32},P, u0::Array{Float32}, pred::Array{Float32}, tspan::Vector{Float32}, t::Float32, dt::Float32, target::Array{Float32})
function loss(l,P, u0, pred, tspan, t, dt, target)
    myprob = ODEProblem{true}(dudt_nn, u0, tspan, p=P)
    pred .= Array(solve(myprob, RK4(), u0 = u0, p = P, saveat=t, dt=dt, verbose=false))
    l .= Float32(sum(abs2, target - pred))
    nothing
end
data = copy(ode_data);
target = copy(ode_data);
l=[T(0.0)];
loss(l,P, u0stacked, data, trange, saveat, dt, target);
l


# Test if the loss can be autodiffed
# [!] dl is called the 'seed' and it has to be marked to be one for correct gradient
l = [T(0.0)];
dl = Enzyme.make_zero(l) .+T(1);
P = ComponentArray{Float32}(f=zeros(T, (n+2,n+2,2)),div=zeros(T,(n+2,n+2)), p=zeros(T,(n+2,n+2)), ft=zeros(T,size(cache_ftemp)), pt=zeros(T,size(cache_ptemp)), lux=zeros(T,(n+2,n+2,2)), θ=copy(θ_node), a_priori=0)
dP = Enzyme.make_zero(P);
du = Enzyme.make_zero(u0stacked);
dd = Enzyme.make_zero(data);
dtarg = Enzyme.make_zero(target);
@timed Enzyme.autodiff(Enzyme.Reverse, loss, DuplicatedNoNeed(l, dl), DuplicatedNoNeed(P, dP), DuplicatedNoNeed(u0stacked, du), DuplicatedNoNeed(data, dd), Const(trange), Const(saveat), Const(dt), DuplicatedNoNeed(target, dtarg))
dP.θ
    
### You can think about reserving a larger stack size for the autodiff using this
#function with_stacksize(f::F, n) where {F<:Function}
#    fetch(schedule(Task(f, n)))
#end
#with_stacksize(2_000_000_000) do
#    Enzyme.autodiff(Enzyme.Reverse, loss, DuplicatedNoNeed(l, dl), DuplicatedNoNeed(P, dP), DuplicatedNoNeed(u0stacked, du) , Const(trange), Const(saveat))
#end


println("Now defining the gradient function")
extra_par = [u0stacked, data, dd, target, dtarg, trange, saveat, dt, du, dP, P];
Textra = typeof(extra_par);
Tth = typeof(P.θ);
#function loss_gradient(G::Tth, θ::Tth, extra_par::Textra) 
function loss_gradient(G, extra_par) 
    u0, data, dd, target, dtarg, trange, saveat, dt, du0, dP, P = extra_par
    # [!] Notice that we are updating P.θ in-place in the loss function
    # Reset gradient to zero
    Enzyme.make_zero!(dP)
    # And remember to pass the seed to the loss funciton with the dual part set to 1
    Enzyme.autodiff(Enzyme.Reverse, loss, DuplicatedNoNeed([T(0)], [T(1)]), DuplicatedNoNeed(P,dP), DuplicatedNoNeed(u0, du0), DuplicatedNoNeed(data, dd) , Const(trange), Const(saveat), Const(dt), DuplicatedNoNeed(target, dtarg))
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
    loss(l,p, u0stacked, data, trange, saveat, dt, target);
    return l
end
callback = function (θ,l; doplot = false)
    println(l)
    return false
end
callback(P, over_loss(P.θ, P))


using SciMLSensitivity, Optimization, OptimizationOptimisers, Optimisers
optf = Optimization.OptimizationFunction((p,u)->over_loss(p,u[end]), grad=(G,p,e)->loss_gradient(G,e))
optprob = Optimization.OptimizationProblem(optf, P.θ, extra_par)


result_e, time_e, alloc_e, gc_e, mem_e = @timed Optimization.solve(optprob,
    OptimizationOptimisers.Adam(0.05),
    callback = callback,
    maxiters = 100)


# Zygote can not be used because we are mutating, then how do we compare?