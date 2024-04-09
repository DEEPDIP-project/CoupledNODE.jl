import Lux: Chain, SkipConnection, Parallel, Upsample, MeanPool

# Returns the function to linearize the input 
function Linearize_in(grids)
    dim = length(grids)
    if dim == 1
        return u -> let u = u
            # make u linear
            u = reshape(u, grids[1].N, size(u)[end])
            u
        end
    elseif dim == 2
        return uv -> let u = uv[1], v = uv[2]
            # make u and v linear
            u = reshape(u, grids[1].N, size(u)[end])
            v = reshape(v, grids[2].N, size(v)[end])
            vcat(u, v)
        end
    elseif dim == 3
        return uvw -> let u = uv[1], v = uv[2], w = uv[3]
            # make u, v and w linear
            u = reshape(u, grids[1].N, size(u)[end])
            v = reshape(v, grids[2].N, size(v)[end])
            w = reshape(w, grids[3].N, size(w)[end])
            vcat(u, v, w)
        end
    end
end

function create_f_CNODE(create_forces, force_params, grids, NNs = nothing;
        is_closed = false, gpu_mode = false)
    # Get the number of equations from the number of grids passed
    dim = length(grids)

    # Check which grids need to be resized
    if dim > 1
        max_dx = maximum([g.dx for g in grids])
        max_dy = maximum([g.dy for g in grids])
        grids_to_rescale = [g.dx != max_dx || g.dy != max_dy ? 1 : 0 for g in grids]
        for (i, needs_rescaling) in enumerate(grids_to_rescale)
            if needs_rescaling == 1
                println("Grid $i needs to be rescaled.")
            end
        end
    end

    # Check if you want to use a GPU
    if gpu_mode
        if !CUDA.functional()
            println("ERROR: no GPU avail")
        end
        grids = cu(grids)
    end

    # Create the forces
    F = create_forces(grids, force_params)

    # Get the functions that will downscale and upscale the field s
    Upscaler = upscale_v(grids, grids_to_rescale, max_dx, max_dy)
    Downscaler = downscale_v(grids, grids_to_rescale, max_dx, max_dy)

    # Check if you want to close the CNODE or not
    if !is_closed
        return Chain(
            Upscaler,
            uv -> let u = uv[1], v = uv[2]
                # Return a tuple of the right hand side of the CNODE
                # remove the placeholder dimension for the channels
                u = reshape(u, grid_for_force.nux, grid_for_force.nuy, size(u, 4))
                v = reshape(v, grid_for_force.nvx, grid_for_force.nvy, size(v, 4))
                F(u, v)
            end,
            Downscaler)
    else
        # Define the NN term that concatenates the output of the NN_u and NN_v
        NN_closure = Parallel(nothing, NN_u, NN_v)
        # And use it to close the CNODE
        return Chain(Upscaler,
            # For the NN I want u and v concatenated in the channel dimension
            uv -> let u = uv[1], v = uv[2]
                cat(u, v, dims = 3)
            end,
            # Apply the right hand side of the CNODE 
            SkipConnection(NN_closure,
                (f_NN, uv) -> let u = uv[:, :, 1, :], v = uv[:, :, 2, :]
                    F(u, v) + f_NN
                end),
            Downscaler)
    end
end

"""
    downscale_v(grid, resize_v=false)
    
Generate the layer that downscales `v` and expects as input a tuple `(u, v)`. 
If `resize_v` is `true`, the layer upscales `v` to twice the target size and applies mean pooling.

Arguments:
- `grid`: Grid object that represents the spatial discretization of the variables `u` and `v`.
- `resize_v`: A boolean indicating whether to resize v.

Returns:
- Function that takes a tuple (u, v) returns the linearized concatenation of `u` and `v`.
"""
function downscale_v(grids, grids_to_rescale, max_dx, max_dy)
    dim = length(grids)
    # Get the explicit function to linearize the input
    linearize_layer = Linearize_in(grids)

    if dim == 1 || !any(grids_to_rescale)
        # There is no need to downscale
        return linearize_layer
    else
        # To downscale we do the following:
        # 1. Create a skippable layer that:
        #       a. Extracts the u-v-w component that needs to be downscaled
        #       b. Upscales it to twice the target size
        #       c. Applies mean pooling
        # 2. Apply a skip connection that gets scaled and unscaled u-v-w components
        # 3. Concatenate the u-v-w components and return them

        # TODO !!! This does nothing for now
        error("Not implemented yet")
        dw_v = Chain(
            # extract only the v component
            uv -> let v = uv[2]
                # to apply upsample you need a 4th dumb dimension to represent the channels/batch
                v = reshape(v, size(v, 1), size(v, 2), size(v, 3), 1)
                v
            end,
            # to downscale we first have to upscale to twice the target size
            Upsample(:trilinear, size = (2 * grid.nvx, 2 * grid.nvy)),
            MeanPool((2, 2)))
        return Chain(SkipConnection(dw_v,
            (v_dw, uv) -> let u = uv[1]
                # make u and v linear
                nbatch = size(u)[end]
                u = reshape(u, grid.Nu, nbatch)
                v = reshape(v_dw, grid.Nv, nbatch)
                # and concatenate
                vcat(u, v)
            end))
    end
end

"""
    upscale_v(grid, resize_v=false)

Generate the layer that upscales `v`.
This function expects as input the linearized concatenation of u and v and returns a tuple (u, v).

Arguments:
- `grid`: Grid object that represents the spatial discretization of the variables `u` and `v`.
- `resize_v`: A boolean indicating whether to resize the v component.

Returns:
- Function that reshapes `u` and `v` on the grid and returns a tuple (u, v)
"""
# Returns the function to concatenate
function Concatenate_in(grids)
    dim = length(grids)
    if dim == 1
        return u -> let u = u
            # add a placeholder dimension for the channels
            u = reshape(u, grids[1].nx, 1, size(u)[end])
            u
        end
    elseif dim == 2
        return uv -> let u = uv[1:(grids[1].N), :], v = uv[(grids[1].N + 1):end, :]
            # reshape u and v on their grid while adding a placeholder dimension for the channels
            u = reshape(u, grid[1].nx, grid[2].ny, 1, size(u, 2))
            v = reshape(v, grid[2].nx, grid[2].ny, 1, size(v, 2))
            (u, v)
        end
    elseif dim == 3
        return uvw -> let u = uvw[1:(grids[1].N), :],
            v = uvw[(grids[1].N + 1):(grids[1].N + grids[2].N), :],
            w = uvw[(grids[1].N + grids[2].N + 1):end, :]
            # reshape u, v and w on their grid while adding a placeholder dimension for the channels
            u = reshape(u, grid[1].nx, grid[1].ny, 1, size(u, 2))
            v = reshape(v, grid[2].nx, grid[2].ny, 1, size(v, 2))
            w = reshape(w, grid[3].nx, grid[3].ny, 1, size(w, 2))
            (u, v, w)
        end
    end
end
# Generate the layer that upscales v
# expects as input the linearized concatenation of u and v
# returns a tuple (u,v)
function upscale_v(grids, grids_to_rescale, max_dx, max_dy)
    dim = length(grids)
    # Get the explicit function to concatenate the input
    concatenate_layer = Concatenate_in(grids)

    if dim == 1 || !any(grids_to_rescale)
        return concatenate_layer
    else

        # TODO!!! This does nothing for now
        error("Not implemented yet")
        up_v = Chain(
            # extract only the v component 
            uv -> let v = uv[(grid.Nu + 1):end, :]
                # reshape v on the grid 
                v = reshape(v, grid.nvx, grid.nvy, 1, size(v, 2))
                v
            end,
            Upsample(:trilinear, size = (grid.nux, grid.nuy)))
        return Chain(SkipConnection(up_v,
            (v_up, uv) -> let u = uv[1:(grid.Nu), :]
                # make u on the grid
                u = reshape(u, grid.nux, grid.nuy, 1, size(u, 2))
                (u, v_up)
            end))
    end
end
