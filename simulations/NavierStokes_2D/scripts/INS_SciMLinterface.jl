##################
# In this script I redefine some functions of the IncompressibleNavierStokes module
# and test if the redefined functions are faster than the original ones
##################

run_test = false
##################
#       psolver
##################
# Here I redefine the psolver function in order to compile it with explicit range functions inside
using SparseArrays, LinearAlgebra
const myzero = T(0)
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

#****** I then compile those parameters into a psolver function
function generate_psolver(viewrange, Ip, fact)
    function psolver(p, f, ftemp, ptemp)
        copyto!(view(ftemp, viewrange), view(view(f, Ip), :))
        ptemp .= fact \ ftemp
        copyto!(view(view(p, Ip), :), eltype(p).(view(ptemp, viewrange)))
        nothing
    end
end
show(err)

##################
#       PBC
##################
# Redefine the apply_bc_p! function in order to comply with Enzyme
using KernelAbstractions
using Enzyme
Enzyme.API.runtimeActivity!(true)
get_bc_p!(p, setup; kwargs...) = let
    (; boundary_conditions, grid, workgroupsize) = setup
    (; dimension, N) = grid
    D = dimension()
    e = Offset{D}()
    B = get_backend(p)
    for β = 1:D
        @assert boundary_conditions[β][1] isa PeriodicBC "Only PeriodicBC implemented"
    end
    
    function bc_p!(p, β; isright)
        @kernel function _bc_a(p, ::Val{β}) where {β}
            I = @index(Global, Cartesian)
            p[I] = p[I+(N[β]-2)*e(β)]
        end
        @kernel function _bc_b(p, ::Val{β}) where {β}
            I = @index(Global, Cartesian)
            p[I+(N[β]-1)*e(β)] = p[I+e(β)]
        end
        ndrange = ntuple(γ -> γ == β ? 1 : N[γ], D)
        if isright
            _bc_b(B, workgroupsize)(p, Val(β); ndrange)
        else
            _bc_a(B, workgroupsize)(p, Val(β); ndrange)
        end
    end

    function bc!(p)
        for β = 1:D
            bc_p!(p, β; isright = false)
            bc_p!(p, β; isright = true)
        end
    end
end

myapply_bc_p! = get_bc_p!(cache_p, setup) 

if run_test
    # Speed test
    @timed for i in 1:10000
        A = rand(Float32,size(cache_p)[1],size(cache_p)[2]);
        IncompressibleNavierStokes.apply_bc_p!(A, 0.0f0, setup);
    end
    @timed for i in 1:10000
        A = rand(Float32,size(cache_p)[1],size(cache_p)[2]);
        myapply_bc_p!(A);
    end

    # Check if the implementation is correct
    for i in 1:10000
        A = rand(Float32,size(cache_p)[1],size(cache_p)[2]) ;
        A0 = copy(A)                   ;
        B = copy(A)                    ;
        IncompressibleNavierStokes.apply_bc_p!(A, T(0), setup)  ;
        myapply_bc_p!(B);
        @assert A == B                  
        @assert A != A0                  
    end

    # Check if it is differentiable
    A = rand(Float32,size(cache_p)[1],size(cache_p)[2]);
    dA = Enzyme.make_zero(A);
    @timed Enzyme.autodiff(Enzyme.Reverse, myapply_bc_p!, Const, DuplicatedNoNeed(A, dA))
end


############# Test a similar thing for the BC on u
get_bc_u!(u, setup; kwargs...) = let
    (; boundary_conditions, grid, workgroupsize) = setup
    (; dimension, N) = grid
    D = dimension()
    e = Offset{D}()
    B = get_backend(u[1])
    for β = 1:D
        @assert boundary_conditions[β][1] isa PeriodicBC "Only PeriodicBC implemented"
    end

    @kernel function _bc_a!(u, ::Val{α}, ::Val{β}) where {α,β}
        I = @index(Global, Cartesian)
        u[α][I] = u[α][I+(N[β]-2)*e(β)]
    end
    @kernel function _bc_b!(u, ::Val{α}, ::Val{β}) where {α,β}
        I = @index(Global, Cartesian)
        u[α][I+(N[β]-1)*e(β)] = u[α][I+e(β)]
    end

    function bc_u!(u, β; isright, )
        ndrange = ntuple(γ -> γ == β ? 1 : N[γ], D)
        if isright
            for α = 1:D
                _bc_b!(B, workgroupsize)(u, Val(α), Val(β); ndrange)
            end
        else
            for α = 1:D
                _bc_a!(B, workgroupsize)(u, Val(α), Val(β); ndrange)
            end
        end
        
    end

    function bc(u)
        for β = 1:D
            bc_u!(u, β; isright = false)
            bc_u!(u, β; isright = true)
        end
    end
end

myapply_bc_u! = get_bc_u!(cache_F, setup)

if run_test
    @timed for i in 1:10000
        A = (rand(Float32,size(cache_p)[1],size(cache_p)[1]),rand(Float32,size(cache_p)[1],size(cache_p)[1]))
        IncompressibleNavierStokes.apply_bc_u!(A, 0.0f0, setup);
    end
    @timed for i in 1:10000
        A = (rand(Float32,size(cache_p)[1],size(cache_p)[1]),rand(Float32,size(cache_p)[1],size(cache_p)[1]))
        myapply_bc_u!(A);
    end

    # Check if the implementation is correct
    for i in 1:1000
        A = (rand(Float32,size(cache_p)[1],size(cache_p)[1]),rand(Float32,size(cache_p)[1],size(cache_p)[1]));
        A0 = (copy(A[1]), copy(A[2])) ;                  ;
        B = (copy(A[1]), copy(A[2]))                    ;
        IncompressibleNavierStokes.apply_bc_u!(A, T(0), setup)  ;
        myapply_bc_u!(B);
        @assert A[1] == B[1]                  
        @assert A[2] == B[2]
        @assert A[1] != A0[1]
        @assert A[2] != A0[2]
    end

    # Check if it is differentiable

    A = (rand(Float32,size(cache_p)[1],size(cache_p)[1]),rand(Float32,size(cache_p)[1],size(cache_p)[1]))
    dA = Enzyme.make_zero(A)
    @timed Enzyme.autodiff(Enzyme.Reverse, myapply_bc_u!, Const, DuplicatedNoNeed(A, dA))
end


###########################################
# **** Test if you can make divergence faster
using KernelAbstractions
using Enzyme
get_divergence!(div, setup) = let
    (; grid, workgroupsize) = setup
    (; Δ, Ip, Np) = grid
    D = length(u)
    e = Offset{D}()
    @kernel function div!(div, u, I0, d, Δ)
        I = @index(Global, Cartesian)
        I = I + I0
#        d[I] .= 0
        for α = 1:D
            #d[I] += (u[α][I] - u[α][I-e(α)]) / Δ[I[α],α]
            d[I] += (u[α][I] - u[α][I-e(α)]) / Δ[α][I[α]]
        end
        div[I] = d[I]
    end
    B = get_backend(div)
    I0 = first(Ip)
    I0 -= oneunit(I0)
    function d!(div, u, d)#, Δ)
        # set the temporary array to zero
        @. d *= 0
        # It requires Δ to be passed from outside to comply with Enzyme
        div!(B, workgroupsize)(div, u, I0, d, Δ; ndrange = Np)
        nothing
    end
end
F = rand(Float32,size(cache_p)[1],size(cache_p)[2])
z = Enzyme.make_zero(F)
u = random_field(setup, T(0))
my_f = get_divergence!(F, setup)
(; grid, Re) = setup
(; Δ, Δu, A) = grid
my_f(F, u, z)#, stack(Δ))

if run_test
    @timed for i in 1:1000
        F0 = rand(Float32,size(cache_p)[1],size(cache_p)[2]);
        F = copy(F0);
        u = random_field(setup, T(0));
        IncompressibleNavierStokes.divergence!(F, u, setup);
        @assert F != F0
    end
    @timed for i in 1:1000
        F = rand(Float32,size(cache_p)[1],size(cache_p)[2]);
        u = random_field(setup, T(0));
        z = Enzyme.make_zero(F)
        my_f(F, u, z)#, stack(Δ));
    end
    # Check if the implementation is correct
    using Statistics
    for i in 1:1000
        F = rand(Float32,size(cache_p)[1],size(cache_p)[2]);
        u = random_field(setup, T(0)) 
        A0 = copy(F)                   ;
        B = copy(F)                    ;
        IncompressibleNavierStokes.divergence!(F, u, setup)  ;
        z = Enzyme.make_zero(F)
        my_f(B, u, z)#, stack(Δ));
        @assert F == B                  
    end

    F = rand(Float32,size(cache_p)[1],size(cache_p)[2]);
    dF = Enzyme.make_zero(F);
    u = random_field(setup, T(0));
    du = Enzyme.make_zero(u);
    d = Enzyme.make_zero(F);
    dd = Enzyme.make_zero(d);
    z = Enzyme.make_zero(F);
    dΔ = Enzyme.make_zero(stack(Δ));
    # Test if it is differentiable
    #@timed Enzyme.autodiff(Enzyme.Reverse, my_f, Const, DuplicatedNoNeed(F, dF), DuplicatedNoNeed(u, du), DuplicatedNoNeed(d, dd), DuplicatedNoNeed(stack(Δ), dΔ))
    @timed Enzyme.autodiff(Enzyme.Reverse, my_f, Const, DuplicatedNoNeed(F, dF), DuplicatedNoNeed(u, du), DuplicatedNoNeed(d, dd))
end


##### Redefine applypressure!
get_applypressure!(u, setup) = let
    (; grid, workgroupsize) = setup
    (; dimension, Δu, Nu, Iu) = grid
    D = dimension()
    e = Offset{D}()
    B = get_backend(u[1])
    @kernel function apply!(u, p, ::Val{α}, I0, Δu) where {α}
        I = @index(Global, Cartesian)
        I = I0 + I
        u[α][I] -= (p[I+e(α)] - p[I]) / Δu[α][I[α]]
    end
    function ap!(u, p)#, Δu)
        for α = 1:D
            I0 = first(Iu[α])
            I0 -= oneunit(I0)
            apply!(B, workgroupsize)(u, p, Val(α), I0, Δu; ndrange = Nu[α])
        end
        nothing
    end
    ap!
end
u = random_field(setup, T(0))
p = rand(T,(n+2,n+2))
myapplypressure! = get_applypressure!(u, setup)
myapplypressure!(u, p)#, Δu)
IncompressibleNavierStokes.applypressure!(u, p, setup)

if run_test
    # Speed test
    @timed for i in 1:1000
        u = random_field(setup, T(0))
        p = rand(T,(n+2,n+2))
        IncompressibleNavierStokes.applypressure!(u, p, setup)
    end
    @timed for i in 1:1000
        u = random_field(setup, T(0))
        p = rand(T,(n+2,n+2))
        myapplypressure!(u, p)#, Δu)
    end

    # Compare with INS
    for i in 1:1000
        u = random_field(setup, T(0))
        p = rand(T,(n+2,n+2))
        u0 = copy.(u)
        IncompressibleNavierStokes.applypressure!(u, p, setup)
        myapplypressure!(u0, p)#, Δu)
        @assert u == u0
    end

    # Check if it is differentiable
    u = random_field(setup, T(0))
    p = rand(T,(n+2,n+2))
    du = Enzyme.make_zero(u)
    dp = Enzyme.make_zero(p)
    dΔu = Enzyme.make_zero(Δu)
    @timed Enzyme.autodiff(Enzyme.Reverse, myapplypressure!, Const, DuplicatedNoNeed(u, du), DuplicatedNoNeed(p, dp))
end


##################Vy
##################Vy
##################Vy
# And now I redefine the momentum! function
get_momentum!(F, u, temp, setup) = let
    (; grid, bodyforce, workgroupsize, Re) = setup
    (; dimension, Nu, Iu, Δ, Δu, A) = grid
    D = dimension()

    function get_convectiondiffusion!(B)
        e = Offset{D}()
        ν = 1 / Re
        (; Δ, Δu, A) = grid
        @kernel function cd!(F, u, ::Val{α}, ::Val{βrange}, I0, Δu, Δ, ν, A) where {α,βrange}
            I = @index(Global, Cartesian)
            I = I + I0
            KernelAbstractions.Extras.LoopInfo.@unroll for β in βrange
                #Δuαβ = α == β ? Δu[:,β] : Δ[:,β]
                Δuαβ = α == β ? Δu[β] : Δ[β]
                uαβ1 = (u[α][I-e(β)] + u[α][I]) / 2
                uαβ2 = (u[α][I] + u[α][I+e(β)]) / 2
                uβα1 =
                    A[β][α][2][I[α]-(α==β)] * u[β][I-e(β)] +
                    A[β][α][1][I[α]+(α!=β)] * u[β][I-e(β)+e(α)]
                uβα2 = A[β][α][2][I[α]] * u[β][I] + A[β][α][1][I[α]+1] * u[β][I+e(α)]
                uαuβ1 = uαβ1 * uβα1
                uαuβ2 = uαβ2 * uβα2
                #∂βuα1 = (u[α][I] - u[α][I-e(β)]) / (β == α ? Δ[I[β],β] : Δu[I[β]-1,β])
                #∂βuα2 = (u[α][I+e(β)] - u[α][I]) / (β == α ? Δ[I[β]+1,β] : Δu[I[β],β])
                ∂βuα1 = (u[α][I] - u[α][I-e(β)]) / (β == α ? Δ[β][I[β]] : Δu[β][I[β]-1])
                ∂βuα2 = (u[α][I+e(β)] - u[α][I]) / (β == α ? Δ[β][I[β]+1] : Δu[β][I[β]])
                F[α][I] += (ν * (∂βuα2 - ∂βuα1) - (uαuβ2 - uαuβ1)) / Δuαβ[I[β]]
            end
        end
        function convdiff!(F, u)#, Δ, Δu)#, ν)#, A)
            for α = 1:D
                I0 = first(Iu[α])
                I0 -= oneunit(I0)
                cd!(B, workgroupsize)(F, u, Val(α), Val(1:D), I0, Δu, Δ, ν, A; ndrange = Nu[α])
            end
            nothing
        end
    end
    function get_bodyforce!(F, u, setup)
        @error "Not implemented"
    end
    convectiondiffusion! = get_convectiondiffusion!(get_backend(F[1]))
    bodyforce! = isnothing(bodyforce) ? (F,u,t)->nothing : get_bodyforce!(F, u, setup)
    gravity! = isnothing(temp) ? (F,u,t)->nothing : INS.gravity!(F, temp, setup)
    function momentum!(F, u, t)#, Δ, Δu)#, ν)#, A, t)
        for α = 1:D
            F[α] .= 0
        end
        convectiondiffusion!(F, u)#, Δ, Δu)#, ν)#, A)
        bodyforce!(F, u, t)
        gravity!(F, temp, setup)
        nothing
    end
end


(; Δ, Δu, A) = grid
ν = 1 / Re

u = random_field(setup, T(0))
F = random_field(setup, T(0))
my_f = get_momentum!(F, u, nothing, setup)
sΔ = stack(Δ)
sΔu = stack(Δu)
my_f(F, u, T(0))#, sΔ, sΔu)#, ν)#, A, T(0))

if run_test
    # Check if it is differentiable
    u = random_field(setup, T(0))
    F = random_field(setup, T(0))
    du = Enzyme.make_zero(u)
    dF = Enzyme.make_zero(F)
    dΔu = Enzyme.make_zero(Δu)
    dΔ = Enzyme.make_zero(Δ)
    dν = Enzyme.make_zero(ν)
    dA = Enzyme.make_zero(A)
    dsΔ = Enzyme.make_zero(sΔ)
    dsΔu = Enzyme.make_zero(sΔu)
#    @timed Enzyme.autodiff(Enzyme.Reverse, my_f, Const, DuplicatedNoNeed(F, dF), DuplicatedNoNeed(u, du), Const(T(0)), DuplicatedNoNeed(sΔ,dsΔ), DuplicatedNoNeed(sΔu, dsΔu))
    @timed Enzyme.autodiff(Enzyme.Reverse, my_f, Const, DuplicatedNoNeed(F, dF), DuplicatedNoNeed(u, du), Const(T(0)))

    @timed for i in 1:1000
        u = random_field(setup, T(0))
        F = random_field(setup, T(0))
        IncompressibleNavierStokes.momentum!(F, u, nothing, T(0), setup)
    end
    @timed for i in 1:1000
        u = random_field(setup, T(0))
        F = random_field(setup, T(0))
        my_f(F, u, T(0))#, sΔ, sΔu)
    end

    # Check if the implementation is correct
    for i in 1:1000
        u = random_field(setup, T(0))
        F = random_field(setup, T(0))
        u0 = copy.(u)
        F0 = copy.(F)
        IncompressibleNavierStokes.momentum!(F, u, nothing, T(0), setup)
        my_f(F0, u, T(0))#, sΔ, sΔu)
        @assert F == F0
    end
end