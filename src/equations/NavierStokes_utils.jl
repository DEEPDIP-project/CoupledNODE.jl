##########################################
### Via IncompressibleNavierStokes.jl ####
##########################################

module NavierStokes

using IncompressibleNavierStokes: IncompressibleNavierStokes as INS
using Lux: Lux

"""
    create_right_hand_side(setup, psolver)

Create right hand side function f(u, p, t) compatible with
the OrdinaryDiffEq ODE solvers. Note that `u` has to be an array.
To convert the tuple `u = (ux, uy)` to an array, use `stack(u)`.
"""
create_right_hand_side(setup, psolver) = function right_hand_side(u, p, t)
    u = eachslice(u; dims = ndims(u))
    u = (u...,)
    u = INS.apply_bc_u(u, t, setup)
    F = INS.momentum(u, nothing, t, setup)
    F = INS.apply_bc_u(F, t, setup; dudt = true)
    PF = INS.project(F, setup; psolver)
    PF = INS.apply_bc_u(PF, t, setup; dudt = true)
    cat(PF[1], PF[2]; dims = 3)
end

"""
    create_right_hand_side_with_closure(setup, psolver, closure, st)

Create right hand side function f(u, p, t) compatible with SciML ODEProblem.
This formulation was the one proved best in Syver's paper i.e. DCF.
`u` has to be an array in the NN style e.g. `[n , n, D]`, with boundary conditions padding.
"""
create_right_hand_side_with_closure(setup, psolver, closure, st) = function right_hand_side(
        u, p, t)
    # not sure if we should keep t as a parameter. t is only necessary for the INS functions when having Dirichlet BCs (time dependent)
    u_INS = NN_padded_to_INS(u, setup)
    u_nopad = NN_padded_to_NN_nopad(u, setup)
    u_for_lux = reshape(u_nopad, size(u_nopad)..., 1)
    u = INS.apply_bc_u(u_INS, t, setup)
    F = INS.momentum(u, nothing, t, setup)
    u_lux = Lux.apply(closure, u_for_lux, p, st)[1]
    FC = sum_NN_nopad_and_INS(u_lux, F, setup)
    FC = INS.apply_bc_u(F, t, setup; dudt = true)
    FP = INS.project(FC, setup; psolver)
    FP = INS.apply_bc_u(FP, t, setup; dudt = true)
    INS_to_NN(FP)
end

"""
    create_right_hand_side_inplace(setup, psolver)

In place version of [`create_right_hand_side`](@ref).
"""
function create_right_hand_side_inplace(setup, psolver)
    (; x, N, dimension) = setup.grid
    D = dimension()
    F = ntuple(α -> similar(x[1], N), D)
    div = similar(x[1], N)
    p = similar(x[1], N)
    function right_hand_side!(dudt, u, params, t)
        u = eachslice(u; dims = ndims(u))
        INS.apply_bc_u!(u, t, setup)
        INS.momentum!(F, u, nothing, t, setup)
        INS.apply_bc_u!(F, t, setup; dudt = true)
        INS.project!(F, setup; psolver, div, p)
        INS.apply_bc_u!(F, t, setup; dudt = true)
        for α in 1:D
            dudt[ntuple(Returns(:), D)..., α] .= F[α]
        end
        dudt
    end
end

"""
    pad_manual(u, D, b_axes, Iu)

Manually pads the input array `u` along each dimension to match the specified axes `b_axes` using the indices `Iu`.

# Arguments
- `u::Array{T}`: The input array to be padded.
- `D::Int`: The number of dimensions to pad.
- `b_axes::Tuple`: A tuple containing the axes of the target array `b_INS`.
- `Iu::NamedTuple`: A named tuple containing the indices for padding with a field `indices` which is a vector of vectors of indices. (stores in IncompressibleNavierStokes.setup)

# Returns
- `Array{T}`: The padded array.
"""
function pad_manual(u, D, b_axes, Iu)
    for i in 1:D
        x1 = (size(u, j) for j in 1:(i - 1))
        x2 = (size(u, j) for j in (i + 1):D)
        presize = first(Iu.indices[i]) - 1
        postsize = last(b_axes[i]) - last(Iu.indices[i])
        pre = similar(u, (x1..., presize, x2...))
        post = similar(u, (x1..., postsize, x2...))
        u = cat(pre, u, post; dims = i)
    end
    return u
end

"""
    sum_NN_nopad_and_INS(a_NN, b_INS, setup)

Sum a neural network output tensor `a_NN` with an incompressible Navier-Stokes tensor `b_INS` after padding `a_NN` to match the dimensions of `b_INS`.

# Arguments
- `a_NN::Array{Float32, 4}`: A tensor in the NN format, with dimensions `(n, n, dim, 1)`.
- `b_INS::Vector{Matrix{Float32}}`: A Tuple of matrices (INS format).
- `setup`: An IncompressibleNavierStokes.jl setup. 

# Returns
- `Tuple{Matrix{Float32}}`: A tuple of matrices (INS format) where each matrix is the sum of the padded `a_NN` tensor and the corresponding `b_INS` matrix.
"""
function sum_NN_nopad_and_INS(a_NN, b_INS, setup)
    (; Iu, dimension) = setup.grid
    D = dimension()
    griddims = ((:) for _ in 1:D)
    b_INS_axes = Tuple(axes(b_INS[i]) for i in eachindex(b_INS))
    Tuple(
        pad_manual(a_NN[griddims..., i, 1], D, b_INS_axes[i], Iu[i]) + b_INS[i]
    for i in 1:length(b_INS)
    )
end

"""
    INS_to_NN(u, setup)

Converts the input velocity field `u` from the IncompressibleNavierStokes.jl style `u[time step]=(ux, uy)`
to a format suitable for neural network training `u[n, n, D, batch]`.

# Arguments
- `u`: Velocity field in INS style.

# Returns
- `u`: Velocity field converted to a tensor format suitable for neural network training.
"""
function INS_to_NN(u)
    cat(u...; dims = ndims(u[1]) + 1)
end

"""
    copy_INS_to_INS_inplace

    helper function to assign in-place to a tuple because I am no julia guru that can one-line this.
"""
function copy_INS_to_INS_inplace!(u_source, u_dest)
    for (u_s, u_d) in zip(u_source, u_dest)
        u_d .= u_s
    end
end

"""
    NN_padded_to_INS(u, setup)

Creates a view of the input velocity field `u` from the neural network data style `u[n, n, D, <optional timesteps>]`
to the IncompressibleNavierStokes.jl style `(ux, uy)`.
If the <optional timesteps> dimensions has size > 1, an error is thrown.

# Arguments
- `u`: Velocity field in NN style.
- `setup`: IncompressibleNavierStokes.jl setup.

# Returns
- `u`: Velocity field view in IncompressibleNavierStokes.jl style.
"""
function NN_padded_to_INS(u, setup)
    (; grid) = setup
    (; dimension) = grid
    D = dimension()
    griddims = ((:) for _ in 1:D)

    if ndims(u) == D + 1
        u_INS = eachslice(u, dims = ndims(u))
        (u_INS...,)
    elseif ndims(u) == D + 2
        if size(u, ndims(u)) != 1
            error("Only a single timeslice is supported")
        end
        Tuple(
            u[griddims..., d, 1]
        for d in 1:size(u, ndims(u) - 1)
        )
    else
        error("Unsupported or non-matching number of dimensions in IO array")
    end
end

"""
    NN_padded_to_NN_nopad(u, setup)

Creates a view of the input velocity field `u` from the neural network data style `u[n, n, D, batch]`
but without boundaries.

# Arguments
- `u`: Velocity field in NN style.
- `setup`: IncompressibleNavierStokes.jl setup.

# Returns
- `u`: Velocity field view without boundaries.
"""
function NN_padded_to_NN_nopad(u, setup)
    (; grid, boundary_conditions) = setup
    (; Iu) = grid
    # Iu has multiple, but similar entries, but there is only one grid. We choose the first one 
    Iu = Iu[1]
    dimdiff = ((:) for _ in 1:(ndims(u) - ndims(Iu)))
    @view u[Iu, dimdiff...]
end

"""
    IO_padded_to_IO_nopad(io, setups)

Creates a view which is similar to the input of the IO arrays, but without boundaries

# Arguments
- `io`::NamedTuple : tuple with all arrays with boundaries.
- `setup`:: IncompressibleNavierStokes.jl setup.

# Returns
- `io`::NamedTuple : tuple with views of all arrays without boundaries.
"""
function IO_padded_to_IO_nopad(io, setups)
    [NamedTuple{keys(io[i])}((NN_padded_to_NN_nopad(x, setups[i]) for x in values(io[i])))
     for i in eachindex(io)]
end

"""
    assert_pad_nopad_similar(io_pad,io_nopad,setup)

Asserts that the values of padded and non-padded arrays are similar (boundaries excluded of course)

# Arguments
-`io_pad`: Padded array
-`io_nopad`: Non-padded array
-`setup`: IncompressibleNavierStokes.jl setup to determine boundary size

# Returns
None

# Throws
- AssertionError
"""
function assert_pad_nopad_similar(io_pad, io_nopad, setup)
    @assert io_nopad == NN_padded_to_NN_nopad(io_pad, setup)
end

"""
    create_io_arrays_posteriori(data, setups)

Main differences between this function and NeuralClosure.create_io_arrays
 - we do not need the commutator error c.
 - we need the time and we keep the initial condition.
 - put time dimension in the end, since SciML also does.

# Returns
A named tuple with fields `u` and `t`.
`u` is a matrix without padding and shape (nless..., D, sample, t)
"""
function create_io_arrays_posteriori(data, setups)
    nsample = length(data)
    ngrid, nfilter = size(data[1].data)
    nt = length(data[1].t) - 1
    T = eltype(data[1].t)
    map(CartesianIndices((ngrid, nfilter))) do I
        ig, ifil = I.I
        (; dimension, N, Iu) = setups[ig].grid
        D = dimension()
        u = zeros(T, N..., D, nsample, nt + 1)
        t = zeros(T, nsample, nt + 1)
        ifield = ntuple(Returns(:), D)
        for is in 1:nsample, it in 1:(nt + 1), α in 1:D
            copyto!(
                view(u, ifield..., α, is, it),
                data[is].data[ig, ifil].u[it][α]
            )
            copyto!(
                view(t, is, :),
                data[is].t
            )
        end
        (; u = u, t = t)
    end
end

"""
    create_dataloader_posteriori(io_array; nunroll=10, device=identity, rng)

Creates a dataloader function for a-posteriori fitting from the given `io_array`.

# Arguments
- `io_array`: A structure containing the data arrays `u` and `t`.
- `nunroll`: The number of time steps to unroll (default is 10).
- `rng`: A random number generator.

# Returns
- A function `dataloader` that, when called, returns a tuple with:
  - `u`: A view into the `u` array of `io_array` for a randomly selected sample and time steps.
  - `t`: The corresponding time steps from the `t` array of `io_array`.

# Notes
- The `io_array.u` array is expected to have dimensions `(n, n, dim, samples, nt)`.
- The `nt` dimension must be greater than or equal to `nunroll`.
- Have only tested in 2D cases.
"""
function create_dataloader_posteriori(io_array; nunroll = 10, rng)
    function dataloader()
        (n, _, dim, samples, nt) = size(io_array.u) # expects that the io_array will be for a i_grid
        @assert nt ≥ nunroll
        istart = rand(rng, 1:(nt - nunroll))
        it = istart:(istart + nunroll)
        isample = rand(rng, 1:samples)
        (; u = view(io_array.u, :, :, :, isample, it), t = io_array.t[isample, it])
    end
end

end # module NavierStokes
