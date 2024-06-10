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

    #unpack = Unpack(grids)
    concatenate = Concatenate(grids)

    if is_closed && isnothing(NNs)
        error("ERROR: NNs should be provided for closed CNODEs")
    end
    if !is_closed && !isnothing(NNs)
        @warn("WARNING: NNs were provided while indicating that the CNODE is not closed")
    end

    # Define the force layer
    apply_force = Force_layer(forces, grids, NNs)

    ## Define the force layer
    #if !is_closed
    #    # If the CNODE is not closed, the force layer is the last layer
    #    apply_force = Force_layer(forces, grids)
    #    if !isnothing(NNs)
    #        @warn("WARNING: NNs were provided while indicating that the CNODE is not closed")
    #    end
    #else
    #    # Define the NN term that concatenates the output of the NNs
    #    if dim == 1
    #        if length(NNs) != 1
    #            error("ERROR: NNs should be a single NN for 1D problems")
    #        end
    #        NN_closure = Parallel(nothing, NNs[1])
    #    elseif dim == 2
    #        if length(NNs) != 2
    #            error("ERROR: NNs should be a tuple of two NNs for 2D problems")
    #        end
    #        # [!] using Parallel with a tuple input means that it gets split to the two NNs!
    #        NN_closure = Parallel(nothing, NNs[1], NNs[2])
    #    elseif dim == 3
    #        if length(NNs) != 3
    #            error("ERROR: NNs should be a tuple of three NNs for 3D problems")
    #        end
    #        NN_closure = Parallel(nothing, NNs[1], NNs[2], NNs[3])
    #    else
    #        error("ERROR: Unsupported number of dimensions: $dim")
    #    end
    #    apply_force = Closure(forces, NN_closure, grids)
    #end

    return Chain(
        #unpack, # redundant (maybe it has to copy back in grid? if change not in place)
        #pre_force,
        apply_force,
        #post_force,
        #concatenate
        )
end

# sciml gets the linear , force gets the grid

#"""
#    Unpack(grids)
#
#Creates a function to unpack the input data from a concatenated list to a tuple.
#
## Arguments
#- `grids`: A vector or tuple containing the grid(s).
#
## Returns
#- A list of the unpacked data (coupled variables)
#"""
function Unpack(grids)
    dim = length(grids)
    if dim == 1
        return u -> let u = u
            grids[1].linear_data[:] .= u[:]
            linear_to_grid(grids[1])
            nothing
        end
    elseif dim == 2
        return uv -> let u = uv[1:(grids[1].N), :], v = uv[(grids[1].N + 1):end, :]
            # reshape u and v on their grid while adding a placeholder dimension for the channels
            grids[1].linear_data[:] .= u[:]
            grids[2].linear_data[:] .= v[:]
            linear_to_grid(grids[1])
            linear_to_grid(grids[2])
            nothing
        end
    elseif dim == 3
        return uvw -> let u = uvw[1:(grids[1].N), :],
            v = uvw[(grids[1].N + 1):(grids[1].N + grids[2].N), :],
            w = uvw[(grids[1].N + grids[2].N + 1):end, :]
            grids[1].linear_data[:] .= u[:]
            grids[2].linear_data[:] .= v[:]
            grids[3].linear_data[:] .= w[:]
            linear_to_grid(grids[1])
            linear_to_grid(grids[2])
            linear_to_grid(grids[3])
            nothing
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
        return u -> 
            # make u linear
            #u = grid_to_linear(grids[1], u)
            #grids[1].grid_data[:] .= u[:]
            #grid_to_linear(grids[1])
            grids[1].linear_data
    elseif dim == 2
        #return uv -> let u = uv[1], v = uv[2]
        return uv -> 
            #grids[1].grid_data[:] .= u[:]
            #grids[2].grid_data[:] .= v[:]
            #grid_to_linear(grids[1])
            #grid_to_linear(grids[2])
            vcat(grids[1].linear_data, grids[2].linear_data)
    end
end

# Applies the right hand side of the CNODE, force F. This is for the not closed problem.
# TODO: make the shape more consistent, such that we can use the same layer for with and without NN
function Force_layer(F, grids, NN_closure=nothing)
    dim = length(F)
    if NN_closure === nothing
        if dim == 1
            return u-> let u=u 
                grids[1].linear_data[:] .= u[:]
                F1 = F[1](grids[1].grid_data)
                return (reshape(view(F1,:,:), grids[1].N, :),)
            end
        elseif dim == 2
            return uv -> let u = uv[1:(grids[1].N), :], v = uv[(grids[1].N + 1):end, :]
                grids[1].linear_data[:] .= u[:]
                grids[2].linear_data[:] .= v[:]
                F1 = F[1](grids[1].grid_data, grids[2].grid_data)
                F2 = F[2](grids[1].grid_data, grids[2].grid_data)
                # make a linear view of Fs and concatenate them
                vcat(reshape(view(F1,:,:,:), grids[1].N, :) , reshape(view(F2,:,:,:), grids[2].N, :))
            end
        elseif dim == 3
            return x-> vcat(F[1](grids[1].grid_data, grids[2].grid_data, grids[3].grid_data), F[2](grids[1].grid_data, grids[2].grid_data, grids[3].grid_data), F[3](grids[1].grid_data, grids[2].grid_data, grids[3].grid_data))
        else
            error("ERROR: Unsupported number of dimensions: $dim")
        end
    else
        if dim == 1
            return u-> let u=u 
                grids[1].linear_data[:] .= u[:]
                F1 = F[1](grids[1].grid_data) + NN_closure[1](grids[1].grid_data)
                return (reshape(view(F1,:,:), grids[1].N, :),)
            end
        elseif dim == 2
            return uv -> let u = uv[1:(grids[1].N), :], v = uv[(grids[1].N + 1):end, :]
                grids[1].linear_data[:] .= u[:]
                grids[2].linear_data[:] .= v[:]
                F1 = F[1](grids[1].grid_data, grids[2].grid_data) + NN_closure[1](grids[1].grid_data, grids[2].grid_data) 
                F2 = F[2](grids[1].grid_data, grids[2].grid_data) + NN_closure[2](grids[1].grid_data, grids[2].grid_data)
                # make a linear view of Fs and concatenate them
                vcat(reshape(view(F1,:,:,:), grids[1].N, :) , reshape(view(F2,:,:,:), grids[2].N, :))
            end
        elseif dim == 3
            return x-> vcat(F[1](grids[1].grid_data, grids[2].grid_data, grids[3].grid_data), F[2](grids[1].grid_data, grids[2].grid_data, grids[3].grid_data), F[3](grids[1].grid_data, grids[2].grid_data, grids[3].grid_data))
        else
            error("ERROR: Unsupported number of dimensions: $dim")
        end
    end
end

## Applies the right hand side of the CNODE, by summing force F and closure NN
#function Closure(F, NN_closure, grids)
#    dim = length(F)
#    if dim == 1
#        return SkipConnection(NN_closure,
#            (f_NN, u) -> (F[1](grids[1].grid_data) .+ f_NN[1],)
#            ; name="Closure")
#    elseif dim == 2
#        return SkipConnection(
#            NN_closure,
#            (f_NN, uv) -> (F[1](grids[1].grid_data, grids[2].grid_data) .+ f_NN[1], F[2](grids[1].grid_data, grids[2].grid_data) .+ f_NN[2])
#            ; name="Closure")
#    else
#        error("ERROR: Unsupported number of dimensions: $dim")
#    end
#end