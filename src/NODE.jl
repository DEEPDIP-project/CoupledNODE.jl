import Lux: Chain, SkipConnection, Parallel, Upsample, MeanPool, identity
import CoupledNODE: linear_to_grid, grid_to_linear

"""
    create_f_CNODE(forces, grids, NNs = nothing; pre_force = identity,
        post_force = identity, is_closed = false)

Create a CoupledNODE (CNODE) function that represents a system of coupled differential equations.

# Arguments
- `forces`: A vector or tuple of functions representing the forces in the system.
- `grids`: A vector or tuple of grids representing the variables in the system.
- `NNs`: (optional) A vector or tuple of neural networks representing the closure terms in the system. Default is `nothing`.
- `pre_force`: (optional) A function to be applied before the force layer. Default is `identity`.
- `post_force`: (optional) A function to be applied after the force layer. Default is `identity`.
- `is_closed`: (optional) A boolean indicating whether the CNODE is closed. Default is `false`.

# Returns
A Chain object representing the consecutive set of operations taking place in the CNODE.
"""
function create_f_CNODE(forces, grids, NNs = nothing; pre_force = identity,
        post_force = identity, is_closed = false)
    # Get the number of equations from the number of grids passed
    dim = length(forces)

    unpack = Unpack(grids)
    concatenate = Concatenate(grids)

    # Define the force layer
    if !is_closed
        # If the CNODE is not closed, the force layer is the last layer
        apply_force = Force_layer(forces)
        if !isnothing(NNs)
            @warn("WARNING: NNs were provided while indicating that the CNODE is not closed")
        end
    else
        # Define the NN term that concatenates the output of the NNs
        if dim == 1
            if length(NNs) != 1
                error("ERROR: NNs should be a single NN for 1D problems")
            end
            NN_closure = Parallel(nothing, NNs[1])
        elseif dim == 2
            if length(NNs) != 2
                error("ERROR: NNs should be a tuple of two NNs for 2D problems")
            end
            # [!] using Parallel with a tuple input means that it gets split to the two NNs!
            NN_closure = Parallel(nothing, NNs[1], NNs[2])
        elseif dim == 3
            if length(NNs) != 3
                error("ERROR: NNs should be a tuple of three NNs for 3D problems")
            end
            NN_closure = Parallel(nothing, NNs[1], NNs[2], NNs[3])
        else
            error("ERROR: Unsupported number of dimensions: $dim")
        end
        apply_force = Closure(forces, NN_closure)
    end

    return Chain(
        unpack,
        pre_force,
        apply_force,
        post_force,
        concatenate)
end

"""
    Unpack(grids)

Creates a function to unpack the input data from a concatenated list to a tuple.

# Arguments
- `grids`: A vector or tuple containing the grid(s).

# Returns
- A list of the unpacked data (coupled variables)
"""
function Unpack(grids)
    dim = length(grids)
    if dim == 1
        return u -> let u = u
            # add a placeholder dimension for the channels
            u = linear_to_grid(grids[1], u)
            u
        end
    elseif dim == 2
        return uv -> let u = uv[1:(grids[1].N), :], v = uv[(grids[1].N + 1):end, :]
            # reshape u and v on their grid while adding a placeholder dimension for the channels
            u = linear_to_grid(grids[1], u)
            v = linear_to_grid(grids[2], v)
            [u, v]
        end
    elseif dim == 3
        return uvw -> let u = uvw[1:(grids[1].N), :],
            v = uvw[(grids[1].N + 1):(grids[1].N + grids[2].N), :],
            w = uvw[(grids[1].N + grids[2].N + 1):end, :]
            # reshape u, v and w on their grid while adding a placeholder dimension for the channels
            u = linear_to_grid(grids[1], u)
            v = linear_to_grid(grids[1], v)
            w = linear_to_grid(grids[1], w)
            [u, v, w]
        end
    end
end

"""
    Concatenate(grids)

Creates a function to concatenate the coupled variables to a single vector.

# Arguments
- `grids`: A vector or tuple containing the grid(s).

# Returns
- A list of concatenated coupled variables.
"""
function Concatenate(grids)
    dim = length(grids)
    if dim == 1
        return u -> let u = u[1]
            # make u linear
            u = grid_to_linear(grids[1], u)
            u
        end
    elseif dim == 2
        return uv -> let u = uv[1], v = uv[2]
            u = grid_to_linear(grids[1], u)
            v = grid_to_linear(grids[2], v)
            vcat(u, v)
        end
    elseif dim == 3
        return uvw -> let u = uv[1], v = uv[2], w = uv[3]
            u = grid_to_linear(grids[1], u)
            v = grid_to_linear(grids[2], v)
            w = grid_to_linear(grids[3], w)
            vcat(u, v, w)
        end
    end
end

# Applies the right hand side of the CNODE, force F. This is for the not closed problem.
# TODO: make the shape more consistent, such that we can use the same layer for with and without NN
function Force_layer(F)
    dim = length(F)
    if dim == 1
        return uv -> let u = uv
            (F[1](u),)
        end
    elseif dim == 2
        return uv -> let u = uv[1], v = uv[2]
            (F[1](u, v), F[2](u, v))
        end
    elseif dim == 3
        return uv -> let u = uv[1], v = uv[2], w = uv[3]
            (F[1](u, v, w), F[2](u, v, w), F[3](u, v, w))
        end
    else
        error("ERROR: Unsupported number of dimensions: $dim")
    end
end

# Applies the right hand side of the CNODE, by summing force F and closure NN
function Closure(F, NN_closure)
    dim = length(F)
    if dim == 1
        return SkipConnection(NN_closure,
            (f_NN, u) -> let u = u
                (F[1](u) .+ f_NN[1],)
            end)
    elseif dim == 2
        return SkipConnection(
            NN_closure,
            (f_NN, uv) -> let u = uv[1], v = uv[2]
                (F[1](u, v) .+ f_NN[1], F[2](u, v) .+ f_NN[2])
            end)
    else
        error("ERROR: Unsupported number of dimensions: $dim")
    end
end
