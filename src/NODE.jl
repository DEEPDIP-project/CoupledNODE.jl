import Lux: Chain, SkipConnection, Parallel, Upsample, MeanPool, identity
import RecursiveArrayTools: ArrayPartition
#import CoupledNODE: linear_to_grid, grid_to_linear
include("./../src/grid.jl")

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

    if is_closed && isnothing(NNs)
        error("ERROR: NNs should be provided for closed CNODEs")
    end
    if !is_closed && !isnothing(NNs)
        @warn("WARNING: NNs were provided while indicating that the CNODE is not closed")
    end

    # Define the force layer
    if !is_closed
        # If the CNODE is not closed, the force layer is the last layer
        apply_force = Force_layer(forces, grids)
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
        apply_force = Force_layer(forces, grids, NN_closure)
    end

    return Chain(
        apply_force
    )
end

# Applies the right hand side of the CNODE, force F. This is for the not closed problem.
# TODO: make the shape more consistent, such that we can use the same layer for with and without NN
function Force_layer(F, grids, NN_closure = nothing)
    dim = length(F)
    if NN_closure === nothing
        if dim == 1
            return u -> F[1](u)
        elseif dim == 2
            # It assumes as input an ArrayPartition
            return uv -> let u = uv.x[1], v = uv.x[2]
                F1 = F[1](u, v)
                F2 = F[2](u, v)
                ArrayPartition(F1, F2)
            end
        else
            error("ERROR: Unsupported number of dimensions: $dim")
        end
    else
        if dim == 1
            return Chain(
            # Compute force + closure 
                SkipConnection(
                NN_closure,
                (f_NN, u) -> F[1](u) + f_NN[1]
                ; name = "Closure"),
            )
        elseif dim == 2
            return Chain(
                SkipConnection(
                NN_closure,
                (f_NN, uv) -> ArrayPartition(F[1](uv.x[1], uv.x[2]) .+ f_NN[1],
                    F[2](uv.x[1], uv.x[2]) .+ f_NN[2])
                ; name = "Closure"),
            )
        else
            error("ERROR: Unsupported number of dimensions: $dim")
        end
    end
end
