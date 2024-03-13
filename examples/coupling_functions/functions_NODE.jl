# Define the NeuralODE using ResNet skip blocks to add the closure
function create_f_NODE(NN, f_u; is_closed=false)
    return Chain(
        SkipConnection(NN, (f_NN, u) -> is_closed ? f_NN + f_u(u) : f_u(u)),
    )
end

# NeuralODE representing the experimental observation
function create_NODE_obs()
    f_o(u) = @. u.*(0.0.-0.8.*log.(u))
    return Chain( 
        u -> f_o(u),
    )
end


function create_f_CNODE(F_u, G_v, grid, NN_u=nothing, NN_v=nothing; is_closed=false)
    # Since I will be upscaling v, I need to define a new grid to pass to the force
    grid_for_force = Grid(grid.dux, grid.duy, grid.nux, grid.nuy, grid.dux, grid.duy, grid.nux, grid.nuy)
    # check if v has to be rescaled or not
    if grid.dux != grid.dvx || grid.duy != grid.dvy
        println("Resizing v to the same grid as u")
        resize_v = true
    else
        println("No need to resize v")
        resize_v = false
    end
    # Get the functions that will downscale and upscale the uv field
    Upscaler = upscale_v(grid, resize_v)
    Downscaler = downscale_v(grid, resize_v)
    
    # Check if you want to close the CNODE or not
    if !is_closed
        return Chain(
            # layer that casts to f32
            Upscaler,
            uv -> let u = uv[1], v = uv[2]
                # Return a tuple of the right hand side of the CNODE
                # remove the placeholder dimension for the channels
                u = reshape(u, grid_for_force.nux, grid_for_force.nuy, size(u, 4))
                v = reshape(v, grid_for_force.nvx, grid_for_force.nvy, size(v, 4))
                (F_u(u, v, grid_for_force), G_v(u, v, grid_for_force))
            end,
            Downscaler,
        )
    else
        # Define the NN term that concatenates the output of the NN_u and NN_v
        NN_closure = Parallel(nothing, NN_u, NN_v)
        # And use it to close the CNODE
        return Chain(
            Upscaler,
            # For the nn I want u and v concatenated in the channel dimension
            uv -> let u = uv[1], v = uv[2]
                cat(u, v, dims=3)
            end,
            # Apply the right hand side of the CNODE 
            SkipConnection(NN_closure, (f_NN, uv) -> let u = uv[:,:,1,:], v = uv[:,:,2,:]
                (F_u(u, v, grid_for_force) + f_NN[1], G_v(u, v, grid_for_force) + f_NN[2])
            end),
            Downscaler,
            )
    end
end

function downscale_v(grid,resize_v=false)
    # Generate the layer that downscales v
    # expects as input a tuple (u,v)
    # returns their linearized concatenation

    if !resize_v
        return Chain(
            uv -> let u = uv[1], v = uv[2]
                # make u and v linear
                u = reshape(u, grid.Nu, size(u)[end])
                v = reshape(v, grid.Nv, size(v)[end])
                # and concatenate
                vcat(u,v) 
            end,
        )
    else
        dw_v = Chain(
                # extract only the v component
                uv -> let v = uv[2]
                    # to apply upsample you need a 4th dumb dimension to represent the channels/batch
                    v = reshape(v, size(v, 1), size(v, 2), size(v, 3), 1)
                    v
                end,
                # to downscale we first have to upscale to twice the target size
                Upsample(:trilinear,size=(2*grid.nvx,2*grid.nvy)),
                MeanPool((2,2)),
            )
        return Chain(SkipConnection(dw_v, (v_dw, uv) -> let u = uv[1]
                    # make u and v linear
                    nbatch = size(u)[end]
                    u = reshape(u, grid.Nu, nbatch)
                    v = reshape(v_dw, grid.Nv, nbatch)
                    # and concatenate
                    vcat(u,v) 
                end),)
    end
end
function upscale_v(grid, resize_v=false)
    # Generate the layer that upscales v
    # expects as input the linearized concatenation of u and v
    # returns a tuple (u,v)

    if !resize_v
        return Chain(
            uv -> let u = uv[1:grid.Nu, :], v = uv[grid.Nu+1:end, :]
                # reshape u and v on the grid
                u = reshape(u, grid.nux, grid.nuy, 1, size(u, 2))
                v = reshape(v, grid.nvx, grid.nvy, 1, size(v, 2))
                (u,v) 
            end,
        )
    else
        up_v = Chain(
                # extract only the v component 
                uv-> let v=uv[grid.Nu+1:end, :]
                    # reshape v on the grid 
                    v = reshape(v, grid.nvx,grid.nvy, 1, size(v,2))
                    v
                end,
                Upsample(:trilinear, size=(grid.nux,grid.nuy)),
        )
        return Chain(SkipConnection(up_v, (v_up, uv) -> let u = uv[1:grid.Nu, :]
                    # make u on the grid
                    u = reshape(u, grid.nux, grid.nuy, 1, size(u, 2))
                    (u,v_up)
                end),)
    end
end

"""
    struct Grid

This object contains the grid information.

Fields:
- `dux::Float64`: The grid spacing in the x-direction for u.
- `duy::Float64`: The grid spacing in the y-direction for u.
- `nux::Int`: The number of grid points in the x-direction for u.
- `nuy::Int`: The number of grid points in the y-direction for u.
- `dvx::Float64`: The grid spacing in the x-direction for v.
- `dvy::Float64`: The grid spacing in the y-direction for v.
- `nvx::Int`: The number of grid points in the x-direction for v.
- `nvy::Int`: The number of grid points in the y-direction for v.
- `Nu::Int`: The total number of elements for u.
- `Nv::Int`: The total number of elements for v.

Constructor:
- `Grid(dux::Float64, duy::Float64, nux::Int, nuy::Int, dvx::Float64, dvy::Float64, nvx::Int, nvy::Int)`: Constructs a `Grid` object with the given grid parameters.
"""
struct Grid
    dux::Float64
    duy::Float64
    nux::Int
    nuy::Int
    dvx::Float64
    dvy::Float64
    nvx::Int
    nvy::Int
    Nu::Int
    Nv::Int

    function Grid(dux::Float64, duy::Float64, nux::Int, nuy::Int, dvx::Float64, dvy::Float64, nvx::Int, nvy::Int)
        Nu = nux * nuy
        Nv = nvx * nvy
        new(dux, duy, nux, nuy, dvx, dvy, nvx, nvy, Nu, Nv)
    end
end