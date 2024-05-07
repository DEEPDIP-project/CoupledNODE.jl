import Lux: Chain, SkipConnection, Parallel, Upsample, MeanPool, identity
import CoupledNODE: linear_to_grid, grid_to_linear

function create_f_CNODE(forces, grids, NNs = nothing, pre_force = identity,
        post_force = identity; is_closed = false)
    # Get the number of equations from the number of grids passed
    dim = length(forces)

    ## Check which grids need to be resized
    #max_dx = maximum([g.dx for g in grids])
    #max_dy = maximum([g.dy for g in grids])
    #if dim > 1
    #    grids_to_rescale = [g.dx != max_dx || g.dy != max_dy ? true : false for g in grids]
    #    for (i, needs_rescaling) in enumerate(grids_to_rescale)
    #        if needs_rescaling
    #            println("Grid $i needs to be rescaled.")
    #        end
    #    end
    #else
    #    grids_to_rescale = [false]
    #end

    unpack = Unpack(grids)
    concatenate = Concatenate(grids)

    # Define the force layer
    if !is_closed
        # If the CNODE is not closed, the force layer is the last layer
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
            # TODO the two NN here take as input the whole thing, is this what you want?
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
Takes as in the linear array containing the variables `u` and `v` and returns a tuple `(u, v)`.
"""
function Unpack(grids)
    dim = length(grids)
    if dim == 1
        return u -> let u = u
            # add a placeholder dimension for the channels
            u = reshape(u, grids[1].nx, prod(size(u)[2:end]))
            u
        end
        # TODO: generalize the fact that u and v can have different dimensions
    elseif dim == 2
        return uv -> let u = uv[1:(grids[1].N), :], v = uv[(grids[1].N + 1):end, :]
            # reshape u and v on their grid while adding a placeholder dimension for the channels
            u = linear_to_grid(grids[1], u)
            v = linear_to_grid(grids[2], v)
            (u, v)
        end
    elseif dim == 3
        # TODO fix this
        return uvw -> let u = uvw[1:(grids[1].N), :],
            v = uvw[(grids[1].N + 1):(grids[1].N + grids[2].N), :],
            w = uvw[(grids[1].N + grids[2].N + 1):end, :]
            # reshape u, v and w on their grid while adding a placeholder dimension for the channels
            u = reshape(u, grids[1].nx, grids[1].ny, size(u)[end])
            v = reshape(v, grids[2].nx, grids[2].ny, size(v)[end])
            w = reshape(w, grids[3].nx, grids[3].ny, size(w)[end])
            (u, v, w)
        end
    end
end

# Returns the function to linearize the input 
function Concatenate(grids)
    dim = length(grids)
    if dim == 1
        return u -> let u = u
            # make u linear
            u = grid_to_linear(grids[1], u)
            u
        end
    elseif dim == 2
        return uv -> let u = uv[1], v = uv[2]
            # make u and v linear
            u = grid_to_linear(grids[1], u)
            v = grid_to_linear(grids[2], v)
            vcat(u, v)
        end
    elseif dim == 3
        # TODO fix this
        return uvw -> let u = uv[1], v = uv[2], w = uv[3]
            # make u, v and w linear
            u = reshape(u, grids[1].N, size(u)[end])
            v = reshape(v, grids[2].N, size(v)[end])
            w = reshape(w, grids[3].N, size(w)[end])
            vcat(u, v, w)
        end
    end
end

#"""
#    downscale_v(grid, resize_v=false)
#    
#Generate the layer that downscales `v` and expects as input a tuple `(u, v)`. 
#If `resize_v` is `true`, the layer upscales `v` to twice the target size and applies mean pooling.
#
#Arguments:
#- `grid`: Grid object that represents the spatial discretization of the variables `u` and `v`.
#- `resize_v`: A boolean indicating whether to resize v.
#
#Returns:
#- Function that takes a tuple (u, v) returns the linearized concatenation of `u` and `v`.
#"""
#function Downscaler(grids, grids_to_rescale, max_dx, max_dy)
#    dim = length(grids)
#    # Get the explicit function to linearize the input
#    linearize_layer = Linearize_in(grids)
#
#    if dim == 1 || !any(grids_to_rescale)
#        # There is no need to downscale
#        return linearize_layer
#    else
#        # To downscale we do the following:
#        # 1. Create a skippable layer that:
#        #       a. Extracts the u-v-w component that needs to be downscaled
#        #       b. Upscales it to twice the target size
#        #       c. Applies mean pooling
#        # 2. Apply a skip connection that gets scaled and unscaled u-v-w components
#        # 3. Concatenate the u-v-w components and return them
#
#        # TODO !!! This does nothing for now
#        error("Not implemented yet")
#        dw_v = Chain(
#            # extract only the v component
#            uv -> let v = uv[2]
#                # to apply upsample you need a 4th dumb dimension to represent the channels/batch
#                v = reshape(v, size(v, 1), size(v, 2), size(v, 3), 1)
#                v
#            end,
#            # to downscale we first have to upscale to twice the target size
#            Upsample(:trilinear, size = (2 * grid.nvx, 2 * grid.nvy)),
#            MeanPool((2, 2)))
#        return Chain(SkipConnection(dw_v,
#            (v_dw, uv) -> let u = uv[1]
#                # make u and v linear
#                nbatch = size(u)[end]
#                u = reshape(u, grid.Nu, nbatch)
#                v = reshape(v_dw, grid.Nv, nbatch)
#                # and concatenate
#                vcat(u, v)
#            end))
#    end
#end

## Generate the layer that upscales v
## expects as input the linearized concatenation of u and v
## returns a tuple (u,v)
#function Upscaler(grids, grids_to_rescale, max_dx, max_dy)
#    dim = length(grids)
#    # Get the explicit function to concatenate the input
#    concatenate_layer = Concatenate_in(grids)
#
#    if dim == 1 || !any(grids_to_rescale)
#        return concatenate_layer
#    else
#
#        # TODO!!! This does nothing for now
#        error("Not implemented yet")
#        up_v = Chain(
#            # extract only the v component 
#            uv -> let v = uv[(grid.Nu + 1):end, :]
#                # reshape v on the grid 
#                v = reshape(v, grid.nvx, grid.nvy, 1, size(v, 2))
#                v
#            end,
#            Upsample(:trilinear, size = (grid.nux, grid.nuy)))
#        return Chain(SkipConnection(up_v,
#            (v_up, uv) -> let u = uv[1:(grid.Nu), :]
#                # make u on the grid
#                u = reshape(u, grid.nux, grid.nuy, 1, size(u, 2))
#                (u, v_up)
#            end))
#    end
#end

# Works only without the NN due to input shape
# TODO: make the shape more consistent, such that we can use the same layer for with and without NN
function Force_layer(F)
    dim = length(F)
    if dim == 1
        return uv -> let u = uv
            # make u linear
            #u = reshape(u, grids[1].N, size(u, 3))
            (F[1](u))
        end
    elseif dim == 2
        return uv -> let u = uv[1], v = uv[2]
            # make u and v linear
            #            u = reshape(u, grids[1].nx, grids[1].ny, size(u, 4))
            #            v = reshape(v, grids[2].nx, grids[2].ny, size(v, 4))
            (F[1](u, v), F[2](u, v))
        end
    elseif dim == 3
        return uv -> let u = uv[1], v = uv[2], w = uv[3]
            # make u, v and w linear
            #u = reshape(u, grids[1].nx, grids[1].ny, grids[1].nz, size(u, 5))
            #v = reshape(v, grids[2].nx, grids[2].ny, grids[2].nz, size(v, 5))
            #w = reshape(w, grids[3].nx, grids[3].ny, grids[3].nz, size(w, 5))
            (F[1](u, v, w), F[2](u, v, w), F[3](u, v, w))
        end
    else
        error("ERROR: Unsupported number of dimensions: $dim")
    end
end

# This function applies the right hand side of the CNODE, by summing force and closure NN
function Closure(F, NN_closure)
    dim = length(F)
    if dim == 1
        return SkipConnection(NN_closure,
            (f_NN, u) -> let u = u
                ## remove the channel dimension
                #u = reshape(u, grids[1].nx, size(u)[end])
                (F[1](u) + f_NN[1])
            end)
    elseif dim == 2
        return SkipConnection(NN_closure,
            (f_NN, uv) -> let u = uv[1], v = uv[2]
                # make u and v linear and remove the channel dimension
                #u = reshape(u, grids[1].nx, grids[1].ny, size(u)[end])
                #v = reshape(v, grids[2].nx, grids[2].ny, size(v)[end])
                (F[1](u, v) + f_NN[1], F[2](u, v) + f_NN[2])
            end)
    else
        error("ERROR: Unsupported number of dimensions: $dim")
    end
end
