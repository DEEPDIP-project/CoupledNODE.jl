##########################################
### Via IncompressibleNavierStokes.jl ####
##########################################
import IncompressibleNavierStokes as INS
import Lux
import NNlib: pad_circular, pad_repeat

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
    INS_to_NN(PF, setup)
end

"""
    create_right_hand_side_with_closure(setup, psolver, closure, st)

Create right hand side function f(u, p, t) compatible with
the OrdinaryDiffEq ODE solvers.
This formulation was the one proved best in Syver's paper i.e. DCF.
`u` has to be an array in the NN style e.g. `[n , n, D, batch]`.
"""
create_right_hand_side_with_closure(setup, psolver, closure, st) = function right_hand_side(
        u, p, t)
    # not sure if we should keep t as a parameter. t is only necessary for the INS functions when having Dirichlet BCs (time dependent)
    u_ins = NN_padded_to_INS(u, setup)
    u = INS.apply_bc_u(u, t, setup)
    F = INS.momentum(u, nothing, t, setup)
    F = F .+
        NN_padded_to_INS(
        Lux.apply(closure, INS_to_NN(u, setup), p, st)[1][:, :, :, 1:1], setup)
    F = INS.apply_bc_u(F, t, setup; dudt = true)
    PF = INS.project(F, setup; psolver)
    PF = INS.apply_bc_u(PF, t, setup; dudt = true)
    INS_to_NN(PF, setup)
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
Right hand side with closure, tries to minimize data copying (more work can be done)
"""
function create_right_hand_side_with_closure_minimal_copy(setup, psolver, closure, st)
    function right_hand_side(u, p, t)
        u_INS = NN_padded_to_INS(u, setup)
        INS.apply_bc_u!(u_INS, t, setup)
        F = INS.momentum(u_INS, nothing, t, setup)
        u_nopad = NN_padded_to_NN_nopad(u, setup)
        u_lux = Lux.apply(closure, u_nopad, p, st)[1][:, :, :, 1:1]
        u_nopad = NN_padded_to_NN_nopad(u, setup)
        u_nopad .= u_lux
        # assert_pad_nopad_similar(u, u_nopad, setup)

        F = F .+ NN_padded_to_INS(u, setup)

        F = INS.apply_bc_u(F, t, setup; dudt = true)
        PF = INS.project(F, setup; psolver)
        PF = INS.apply_bc_u(PF, t, setup; dudt = true)
        cat(PF[1], PF[2]; dims = 3)
    end
end

"""
    INS_to_NN(u, setup)

Converts the input velocity field `u` from the IncompressibleNavierStokes.jl style `u[time step]=(ux, uy)`
to a format suitable for neural network training `u[n, n, D, batch]`.

# Arguments
- `u`: Velocity field in INS style.
- `setup`: IncompressibleNavierStokes.jl setup.

# Returns
- `u`: Velocity field converted to a tensor format suitable for neural network training.
"""
function INS_to_NN(u, setup)
    (; dimension, Iu) = setup.grid
    D = dimension()
    if D == 2
        u = cat(u[1][Iu[1]], u[2][Iu[2]]; dims = 3) # From tuple to tensor
        u = reshape(u, size(u)..., 1)
    elseif D == 3
        u = cat(u[1][Iu[1]], u[2][Iu[2]], u[3][Iu[3]]; dims = 4)
        u = reshape(u, size(u)..., 1) # One sample
    else
        error("Unsupported dimension: $D. Only 2D and 3D are supported.")
    end
end

"""
    NN_to_INS(u, setup)

Converts the input velocity field `u` from the neural network data style `u[n, n, D, batch]`
to the IncompressibleNavierStokes.jl style `u[time step]=(ux, uy)`.

# Arguments
- `u`: Velocity field in NN style.
- `setup`: IncompressibleNavierStokes.jl setup.

# Returns
- `u`: Velocity field converted to IncompressibleNavierStokes.jl style.
"""

function NN_padded_to_INS(u, setup)
    (; grid, boundary_conditions) = setup
    (; dimension) = grid
    D = dimension()
    # Not really sure about the necessity for the distinction. 
    # We just want a place holder for the boundaries and they will in any case be re-calculated via INS.apply_bc_u!
    if D == 2
        (@view(u[:, :, 1, 1]), @view(u[:, :, 1, 1]))
    elseif D == 3
        (@view(u[:, :, :, 1, 1]), @view(u[:, :, :, 2, 1]), @view(u[:, :, :, 3, 1]))
    else
        error("Unsupported dimension: $D. Only 2D and 3D are supported.")
    end
end

function NN_padded_to_NN_nopad(u, setup)
    (; grid, boundary_conditions) = setup
    (; dimension) = grid
    D = dimension()
    # Not really sure about the necessity for the distinction. 
    # We just want a place holder for the boundaries and they will in any case be re-calculated via INS.apply_bc_u!
    if D == 2
        x, y, _... = size(u)
        @view u[2:(x - 1), 2:(y - 1), :, :]
    elseif D == 3
        x, y, z, _, _ = size(u)
        @view u[2:(x - 1), 2:(y - 1), 2:(z - 1), :, :]
    else
        error("Unsupported dimension: $D. Only 2D and 3D are supported.")
    end
end

function IO_padded_to_IO_nopad(io, setups)
    [NamedTuple{keys(io[i])}((NN_padded_to_NN_nopad(x, setups[i]) for x in values(io[i])))
     for i in 1:length(io)]
end

function assert_pad_nopad_similar(io_pad, io_nopad, setup)
    @assert io_nopad == NN_padded_to_NN_nopad(io_pad, setup)
end
