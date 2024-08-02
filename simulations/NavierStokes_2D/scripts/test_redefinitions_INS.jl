##################
# In this script I redefine some functions of the IncompressibleNavierStokes module
# and test if the redefined functions are faster than the original ones
##################

##################
#       psolver
##################
# Here I redefine the psolver function in order to compile it with explicit range functions inside
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
# In general I have noticed that the PBC function show the following:
#   INS can be differentiated a priori but not a posteriori
#   my custom implementation down here can be differentiated only a posteriori 
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

# Speed test
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
    IncompressibleNavierStokes.apply_bc_p!(A, T(0), setup)  ;
    myapply_bc_p!(B, T(0), setup);
    @assert A ≈ B                  
end


############# Test a similar thing for the BC on u
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
    IncompressibleNavierStokes.apply_bc_u!(A, T(0), setup)  ;
    myapply_bc_u!(B, T(0), setup);
    @assert A[1] ≈ B[1]                  
    @assert A[2] ≈ B[2]
end


###########################################
# **** Test if you can make divergence faster
using KernelAbstractions
using Enzyme
function get_divergence!(div, setup)
    (; grid, workgroupsize) = setup
    (; Δ, Ip, Np) = grid
    D = length(u)
    e = Offset{D}()
    @kernel function div!(div, u, I0, d, Δ)
        I = @index(Global, Cartesian)
        I = I + I0
        d[I] *= 0
        for α = 1:D
            d[I] += (u[α][I] - u[α][I-e(α)]) / Δ[I[α],α]
        end
        div[I] = d[I]
    end
    B = get_backend(div)
    I0 = first(Ip)
    I0 -= oneunit(I0)
    function d!(div, u, d, Δ)
        # It requires Δ to be passed from outside to comply with Enzyme
        div!(B, workgroupsize)(div, u, I0, d, Δ; ndrange = Np)
        synchronize(B)
        nothing
    end
end
A = rand(Float32,size(cache_p)[1],size(cache_p)[2])
z = Enzyme.make_zero(A)
u = random_field(setup, T(0))
my_f = get_divergence!(A, setup)
(; grid) = setup
(; Δ, Δu) = grid
my_f(A, u, z, stack(Δ))

@timed for i in 1:1000
    A0 = rand(Float32,size(cache_p)[1],size(cache_p)[2]);
    A = copy(A0);
    u = random_field(setup, T(0));
    IncompressibleNavierStokes.divergence!(A, u, setup);
    @assert A != A0
end
@timed for i in 1:1000
    A = rand(Float32,size(cache_p)[1],size(cache_p)[2]);
    u = random_field(setup, T(0));
    z = Enzyme.make_zero(A)
    my_f(A, u, z, stack(Δ));
end
# Check if the implementation is correct
using Statistics
for i in 1:1000
    A = rand(Float32,size(cache_p)[1],size(cache_p)[2]);
    u = random_field(setup, T(0)) .* rand()
    A0 = copy(A)                   ;
    B = copy(A)                    ;
    IncompressibleNavierStokes.divergence!(A, u, setup)  ;
    z = Enzyme.make_zero(A)
    my_f(B, u, z, stack(Δ));
    @assert A ≈ B                  
end

A = rand(Float32,size(cache_p)[1],size(cache_p)[2]);
dA = Enzyme.make_zero(A);
u = random_field(setup, T(0));
du = Enzyme.make_zero(u);
d = Enzyme.make_zero(A);
dd = Enzyme.make_zero(d);
z = Enzyme.make_zero(A);
dΔ = Enzyme.make_zero(stack(Δ));
# Test if it is differentiable
#@timed Enzyme.autodiff(Enzyme.Reverse, my_f, Const, DuplicatedNoNeed(A, dA), DuplicatedNoNeed(u, du), DuplicatedNoNeed(d, dd), DuplicatedNoNeed(stack(Δ), dΔ))


##### Redefine applypressure!
function get_applypressure!(u, setup)
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
        synchronize(B)
        nothing
    end
    ap!
end
u = random_field(setup, T(0))
p = rand(T,(n+2,n+2))
myapplypressure! = get_applypressure!(u, setup)
myapplypressure!(u, p)#, Δu)
IncompressibleNavierStokes.applypressure!(u, p, setup)

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
#@timed Enzyme.autodiff(Enzyme.Reverse, myapplypressure!, Const, DuplicatedNoNeed(u, du), DuplicatedNoNeed(p, dp))


##################Vy
##################Vy
##################Vy
# And now I redefine the momentum! function

function momentum!(F, u, temp, t, setup)
    (; grid, closure_model, temperature) = setup
    (; dimension) = grid
    D = dimension()
    for α = 1:D
        F[α] .= 0
    end
    # diffusion!(F, u, setup)
    # convection!(F, u, setup)
    convectiondiffusion!(F, u, setup)
    bodyforce!(F, u, t, setup)
    isnothing(temp) || gravity!(F, temp, setup)
    F
end