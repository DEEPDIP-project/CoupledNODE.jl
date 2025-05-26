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
function create_right_hand_side(setup, psolver)
    function right_hand_side(u, p, t)
        @error "Deprecated: look at the following link: https://agdestein.github.io/IncompressibleNavierStokes.jl/dev/manual/sciml "
        u = INS.apply_bc_u(u, t, setup)
        F = INS.momentum(u, nothing, t, setup)
        F = INS.apply_bc_u(F, t, setup; dudt = true)
        PF = INS.project(F, setup; psolver)
    end
end

"""
    create_right_hand_side_with_closure(setup, psolver, closure, st)

Create right hand side function f(u, p, t) compatible with SciML ODEProblem.
This formulation was the one proved best in Syver's paper i.e. DCF.
`u` has to be an array in the NN style e.g. `[n , n, D]`, with boundary conditions padding.
"""
function create_right_hand_side_with_closure(setup, psolver, closure, st)
    function right_hand_side(
            u, p, t)
        u = INS.apply_bc_u(u, t, setup)
        F = INS.momentum(u, nothing, t, setup)
        u_lux = u[axes(u)..., 1:1] # Add batch dimension
        u_lux = Lux.apply(closure, u_lux, p, st)[1]
        u_lux = u_lux[axes(u)..., 1] # Remove batch dimension
        FC = F .+ u_lux
        FC = INS.apply_bc_u(FC, t, setup; dudt = true)
        FP = INS.project(FC, setup; psolver)
        return FP
    end
end

"""
    create_right_hand_side_inplace(setup, psolver)

In place version of [`create_right_hand_side`](@ref).
"""
function create_right_hand_side_inplace(setup, psolver)
    p = scalarfield(setup)
    function right_hand_side!(dudt, u, params, t)
        # [!]*** be careful to not touch u in this function!
        temp_vector = copy(u)
        INS.apply_bc_u!(temp_vector, t, setup)
        INS.momentum!(dudt, temp_vector, nothing, t, setup)
        INS.apply_bc_u!(dudt, t, setup; dudt = true)
        INS.project!(dudt, setup; psolver, p)
        return nothing
    end
end
function create_right_hand_side_with_closure_inplace(setup, psolver, closure, st)
    p = scalarfield(setup)
    temp_vector = vectorfield(setup)
    u_lux = temp_vector[axes(u)..., 1:1] # Add batch dimension
    function right_hand_side!(dudt, u, params, t)
        # [!]*** be careful to not touch u in this function!
        temp_vector .= copy(u)
        INS.apply_bc_u!(temp_vector, t, setup)
        INS.momentum!(dudt, temp_vector, nothing, t, setup)
        u_lux .= temp_vector[axes(u)..., 1:1]
        u_lux = Lux.apply(closure, u_lux, params, st)[1]
        #u_lux = u_lux[axes(u)..., 1] # Remove batch dimension
        dudt .+= u_lux[axes(u)..., 1] # Remove batch dimension
        INS.apply_bc_u!(dudt, t, setup; dudt = true)
        INS.project!(dudt, setup; psolver, p)
        return nothing
    end
end

"""
    create_io_arrays_priori(data, setups)

Create ``(\\bar{u}, c)`` pairs for training.
# Returns
A named tuple with fields `u` and `c`. (without boundary conditions padding)
"""
function INS_create_io_arrays_priori(data, setups)
    # This is a reference function that creates the io_arrays for the a-priori
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
        for is in 1:nsample
            copyto!(
                view(u,(ifield...),:,:,is),
                data[is][ig, ifil].u[Iu[1], :, :]
            )
            copyto!(
                view(c,(ifield...),:,:,is),
                data[is][ig, ifil].c[Iu[1], :, :]
            )
        end
        (; u = reshape(u, (N .- 2)..., D, :), c = reshape(c, (N .- 2)..., D, :))
    end
end
function create_io_arrays_priori(data, setup, device = identity)
    # This is a reference function that creates the io_arrays for the a-priori
    nsample = length(data)
    @info "Creating io_arrays for a-priori. I find $(nsample) samples."
    (; dimension, N, Iu) = setup.grid
    nt = length(data[1].t) - 1
    T = eltype(data[1].t[1])
    (; dimension, N, Iu) = setup.grid
    D = dimension()
    u = zeros(T, (N .- 2)..., D, nt + 1, nsample)
    c = zeros(T, (N .- 2)..., D, nt + 1, nsample)
    ifield = ntuple(Returns(:), D)
    for is in 1:nsample
        copyto!(
            view(u,(ifield...),:,:,is),
            data[is].u[Iu[1], :, :]
        )
        copyto!(
            view(c,(ifield...),:,:,is),
            data[is].c[Iu[1], :, :]
        )
    end
    (; u = device(reshape(u, (N .- 2)..., D, :)), c = device(reshape(c, (N .- 2)..., D, :)))
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
function create_io_arrays_posteriori(data, setup, device = identity)
    nsample = length(data)
    nt = length(data[1].t) - 1
    T = eltype(data[1].t[1])
    (; dimension, N) = setup.grid
    D = dimension()
    u = zeros(T, N..., D, nsample, nt + 1)
    t = zeros(T, nsample, nt + 1)
    ifield = ntuple(Returns(:), D)
    for is in 1:nsample
        copyto!(
            view(u,(ifield...),:,is,:),
            data[is].u[ifield..., :, :]
        )
        copyto!(
            view(t, is, :),
            data[is].t[:]
        )
    end
    (; u = device(u), t = t)
end
function INS_create_io_arrays_posteriori(data, setups, device = identity)
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
                view(u,(ifield...),:,is,:),
                data[is][ig, ifil].u[ifield..., :, :]
            )
        end
        for is in 1:nsample
            copyto!(
                view(t, is, :),
                data[is][1, 1].t
            )
        end
        (; u = device(u), t = t)
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
function create_dataloader_prior(io_array; batchsize = 50, device = identity, rng)
    function dataloader()
        x, y = io_array
        nsample = size(x)[end]
        d = ndims(x)
        i = sort(shuffle(rng, 1:nsample)[1:batchsize])
        xuse = device(selectdim(x, d, i))
        yuse = device(selectdim(y, d, i))
        xuse, yuse
    end
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
function create_dataloader_posteriori(
        io_array; nunroll = 10, nsamples = 1, device = identity, rng)
    function dataloader()
        (n..., dim, _, _) = axes(io_array.u)
        (_..., samples, nt) = size(io_array.u)

        @assert nt ≥ nunroll
        @assert nsamples ≤ samples "Requested nsamples ($nsamples) exceeds available samples ($samples)"

        # Select starting point for unrolling
        istart = rand(rng, 1:(nt - nunroll))
        it = istart:(istart + nunroll)

        # Select multiple samples
        isamples = rand(rng, 1:samples, nsamples)

        # Use views and batch data movement
        u = view(io_array.u, n..., dim, isamples, it)
        t = view(io_array.t, isamples, it)

        if device != identity
            u = device(copy(u))
            t = device(copy(t))
        end

        (; u = u, t = t)
    end
end
