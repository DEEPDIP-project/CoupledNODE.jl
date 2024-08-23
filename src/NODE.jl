import Lux: Chain, SkipConnection, Parallel, Upsample, MeanPool, identity
import RecursiveArrayTools: ArrayPartition
#import CoupledNODE: linear_to_grid, grid_to_linear
include("./../src/grid.jl")

"""
Create a CoupledNODE (CNODE) function that represents a system of coupled differential equations.

# Returns
A Chain object representing the consecutive set of operations taking place in the CNODE.
"""
function create_f_CNODE(forces, grids = nothing, NNs = nothing; pre_force = identity,
        post_force = identity, is_closed = false, only_closure = false)
    # Get the number of equations from the number of grids passed
    dim = length(forces)

    if is_closed && isnothing(NNs)
        error("ERROR: NNs should be provided for closed CNODEs")
    end
    if !is_closed && !isnothing(NNs)
        @warn("WARNING: NNs were provided while indicating that the CNODE is not closed")
    end
    # TODO: remove this warning when updating the code to not pass grids
    if grids !== nothing
        @warn("WARNING: grids are not used in the current implementation, so you can remove them")
    end

    if only_closure
        if dim > 1
            error("ERROR: only_closure is only supported for 1D problems")
        end
        if is_closed
            error("ERROR: only_closure is not supported for closed CNODEs")
        end
        if length(NNs) != 1
            error("ERROR: only_closure is only supported for a single NN")
        end
        println("Creating a CNODE with only the closure term")
        return Chain(NNs[1])
    end

    # Define the force layer
    if !is_closed
        apply_force = Force_layer(forces)
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
        apply_force = Force_layer(forces, NN_closure)
    end

    return Chain(
        apply_force
    )
end

# Applies the right hand side of the CNODE, force F. This is for the not closed problem.
# TODO: make the shape more consistent, such that we can use the same layer for with and without NN
function Force_layer(F, NN_closure = nothing)
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
