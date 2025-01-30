##########################################
### Via IncompressibleNavierStokes.jl ####
##########################################

using IncompressibleNavierStokes: IncompressibleNavierStokes as INS

"""
    create_right_hand_side(setup, psolver)

Create right hand side function f(u, p, t) compatible with
the OrdinaryDiffEq ODE solvers. Note that `u` has to be an array.
To convert the tuple `u = (ux, uy)` to an array, use `stack(u)`.
"""
create_right_hand_side(setup, psolver) = function right_hand_side(u, p, t)
    u = INS.apply_bc_u(u, t, setup)
    F = INS.momentum(u, nothing, t, setup)
    F = INS.apply_bc_u(F, t, setup; dudt = true)
    PF = INS.project(F, setup; psolver)
    PF = INS.apply_bc_u(PF, t, setup; dudt = true)
    return PF
end

"""
    create_right_hand_side_with_closure(setup, psolver, closure, st)

Create right hand side function f(u, p, t) compatible with SciML ODEProblem.
This formulation was the one proved best in Syver's paper i.e. DCF.
`u` has to be an array in the NN style e.g. `[n , n, D]`, with boundary conditions padding.
"""
create_right_hand_side_with_closure(setup, psolver, closure, st) = function right_hand_side(
        u, p, t)
    u = INS.apply_bc_u(u, t, setup)
    F = INS.momentum(u, nothing, t, setup)
    u_lux = u[axes(u)..., 1:1] # Add batch dimension
    u_lux = Lux.apply(closure, u_lux, p, st)[1]
    u_lux = u_lux[axes(u)..., 1] # Remove batch dimension
    FC = F .+ u_lux
    FC = INS.apply_bc_u(FC, t, setup; dudt = true)
    FP = INS.project(FC, setup; psolver)
    FP = INS.apply_bc_u(FP, t, setup; dudt = true)
    return FP
end

"""
    create_right_hand_side_inplace(setup, psolver)

In place version of [`create_right_hand_side`](@ref).
"""
function create_right_hand_side_inplace(setup, psolver)
    (; x, N, dimension) = setup.grid
    D = dimension()
    F = similar(u)
    div = similar(x[1], N)
    p = similar(x[1], N)
    function right_hand_side!(dudt, u, params, t)
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
    create_io_arrays_priori(data, setups)

Create ``(\\bar{u}, c)`` pairs for training.
# Returns
A named tuple with fields `u` and `c`. (without boundary conditions padding)
"""
function create_io_arrays_priori(data, setups)
    nsample = length(data)
    ngrid, nfilter = size(data[1])
    nt = length(data[1][1].t) - 1
    T = eltype(data[1][1].t)
    map(CartesianIndices((ngrid, nfilter))) do I
        ig, ifil = I.I
        (; dimension, N, Iu) = setups[ig].grid
        D = dimension()
        u = zeros(T, (N .- 2)..., D, nt + 1, nsample)
        c = zeros(T, (N .- 2)..., D, nt + 1, nsample)
        ifield = ntuple(Returns(:), D)
        for is in 1:nsample, it in 1:(nt + 1)
            copyto!(
                view(u, ifield..., :, :, is),
                data[is][ig, ifil].u[Iu[1], :, :]
            )
            copyto!(
                view(c, ifield..., :, :, is),
                data[is][ig, ifil].c[Iu[1], :, :]
            )
        end
        (; u = reshape(u, (N .- 2)..., D, :), c = reshape(c, (N .- 2)..., D, :))
    end
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
    ngrid, nfilter = size(data[1])
    nt = length(data[1][1].t) - 1
    T = eltype(data[1][1].t)
    map(CartesianIndices((ngrid, nfilter))) do I
        ig, ifil = I.I
        (; dimension, N, Iu) = setups[ig].grid
        D = dimension()
        u = zeros(T, N..., D, nsample, nt + 1)
        t = zeros(T, nsample, nt + 1)
        ifield = ntuple(Returns(:), D)
        for is in 1:nsample
            copyto!(
                view(u, ifield..., :, is, :),
                data[is][ig, ifil].u[ifield..., :, :]
            )
        end
        for is in 1:nsample
            copyto!(
                view(t, is, :),
                data[is][1, 1].t
            )
        end
        (; u = u, t = t)
    end
end

"""
    create_dataloader_prior(io_array; nunroll=10, device=identity, rng)

    Create dataloader that uses a batch of `batchsize` random samples from
`data` at each evaluation.

# Arguments
- `io_array`: An named tuple with the data arrays `u` and `c`.
- `batchsize`: The number of samples to use in each batch.
- `device`: A function to move the data to the desired device (e.g. gpu).

# Returns
- A function `dataloader` that, when called, returns a tuple with:
  - `u`: bar_u from `io_array` (input to NN)
  - `c`: commutator error from `io_array` (label)
"""
create_dataloader_prior(io_array; batchsize = 50, device = identity, rng) = function dataloader()
    x, y = io_array
    nsample = size(x)[end]
    d = ndims(x)
    i = sort(shuffle(rng, 1:nsample)[1:batchsize])
    @warn "Dataloader is using device: $device"
    xuse = device(Array(selectdim(x, d, i)))
    yuse = device(Array(selectdim(y, d, i)))
    xuse, yuse
end

"""
    create_dataloader_posteriori(io_array; nunroll=10, device=identity, rng)

Creates a dataloader function for a-posteriori fitting from the given `io_array`.

# Arguments
- `io_array`: A structure containing the data arrays `u` and `t`.
- The `io_array.u` array is expected to have dimensions `(nx, ny... , dim, samples, nt)`.
- `nunroll`: The number of time steps to unroll (default is 10).
- `rng`: A random number generator.

# Returns
- A function `dataloader` that, when called, returns a tuple with:
  - `u`: A view into the `u` array of `io_array` for a randomly selected sample and time steps.
  - `t`: The corresponding time steps from the `t` array of `io_array`.

# Notes
- The `nt` dimension must be greater than or equal to `nunroll`.
- Have only tested in 2D cases.
- It assumes that the data are loaded in batches of size 1
"""
function create_dataloader_posteriori(io_array; nunroll = 10, device = identity, rng)
    function dataloader()
        (n..., dim, _, _) = axes(io_array.u) # expects that the io_array will be for a i_grid
        (_..., samples, nt) = size(io_array.u)

        @assert nt ≥ nunroll
        # select starting point for unrolling
        istart = rand(rng, 1:(nt - nunroll))
        it = istart:(istart + nunroll)
        # select the sample
        isample = rand(rng, 1:samples)
        (; u = view(io_array.u, n..., dim, isample, it), t = io_array.t[isample, it])
    end
end
