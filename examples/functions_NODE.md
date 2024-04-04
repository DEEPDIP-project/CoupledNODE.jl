```julia
import Lux: Chain, SkipConnection, Parallel, Upsample, MeanPool

"""
    create_f_NODE(NN, f_u; is_closed=false)

Create a Neural ODE (NODE) using ResNet skip blocks to add the closure.
```

Arguments

```julia
- `NN`: The neural network model.
- `f_u`: The closure function.
- `is_closed`: A boolean indicating whether to add the closure or not. Default is `false`.
```

Returns

```julia
- The created Neural ODE (NODE) model.
"""
function create_f_NODE(NN, f_u; is_closed = false)
    return Chain(SkipConnection(NN, (f_NN, u) -> is_closed ? f_NN + f_u(u) : f_u(u)))
end

"""
    create_f_CNODE(F_u, G_v, grid, NN_u=nothing, NN_v=nothing; is_closed=false)

Create a neural network model for the Coupled Neural ODE (CNODE) approach.
```

Arguments

```julia
- `F_u`: Function that defines the right-hand side of the CNODE for the variable `u`.
- `G_v`: Function that defines the right-hand side of the CNODE for the variable `v`.
- `grid`: Grid object that represents the spatial discretization of the variables `u` and `v`.
- `NN_u` (optional): Neural network model for the variable `u`. Default is `nothing`.
- `NN_v` (optional): Neural network model for the variable `v`. Default is `nothing`.
- `is_closed` (optional): Boolean indicating whether the CNODE is closed or not. Default is `false`.
```

Returns

```julia
- The created CNODE model.
"""
function create_f_CNODE(create_forces, force_params, grid, NN_u = nothing,
        NN_v = nothing; is_closed = false, gpu_mode = false)
```

Since I will be upscaling v, I need to define a new grid to pass to the force

```julia
    grid_for_force = Grid(grid.dux,
        grid.duy,
        grid.nux,
        grid.nuy,
        grid.dux,
        grid.duy,
        grid.nux,
        grid.nuy,
        convert_to_float32 = true)
    if gpu_mode
        if !CUDA.functional()
            println("ERROR: no GPU avail")
        end
        grid_for_force = cu(grid_for_force)
    end
    F = create_forces(grid_for_force, force_params)
```

check if v has to be rescaled or not

```julia
    if grid.dux != grid.dvx || grid.duy != grid.dvy
        println("Resizing v to the same grid as u")
        resize_v = true
    else
        println("No need to resize v")
        resize_v = false
    end
```

Get the functions that will downscale and upscale the uv field

```julia
    Upscaler = upscale_v(grid, resize_v)
    Downscaler = downscale_v(grid, resize_v)
```

Check if you want to close the CNODE or not

```julia
    if !is_closed
        return Chain(
            Upscaler,
            uv -> let u = uv[1], v = uv[2]
```

Return a tuple of the right hand side of the CNODE
remove the placeholder dimension for the channels

```julia
                u = reshape(u, grid_for_force.nux, grid_for_force.nuy, size(u, 4))
                v = reshape(v, grid_for_force.nvx, grid_for_force.nvy, size(v, 4))
                F(u, v)
            end,
            Downscaler)
    else
```

Define the NN term that concatenates the output of the NN_u and NN_v

```julia
        NN_closure = Parallel(nothing, NN_u, NN_v)
```

And use it to close the CNODE

```julia
        return Chain(Upscaler,
```

For the NN I want u and v concatenated in the channel dimension

```julia
            uv -> let u = uv[1], v = uv[2]
                cat(u, v, dims = 3)
            end,
```

Apply the right hand side of the CNODE

```julia
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
function downscale_v(grid, resize_v = false)
    if !resize_v
        return Chain(uv -> let u = uv[1], v = uv[2]
            #println("in down")
            #println(typeof(u))
```

make u and v linear

```julia
            u = reshape(u, grid.Nu, size(u)[end])
            v = reshape(v, grid.Nv, size(v)[end])
            #println("out down")
            #println(typeof(u))
            #println(typeof(vcat(u, v)))
```

and concatenate

```julia
            vcat(u, v)
        end)
    else
        dw_v = Chain(
```

extract only the v component

```julia
            uv -> let v = uv[2]
```

to apply upsample you need a 4th dumb dimension to represent the channels/batch

```julia
                v = reshape(v, size(v, 1), size(v, 2), size(v, 3), 1)
                v
            end,
```

to downscale we first have to upscale to twice the target size

```julia
            Upsample(:trilinear, size = (2 * grid.nvx, 2 * grid.nvy)),
            MeanPool((2, 2)))
        return Chain(SkipConnection(dw_v,
            (v_dw, uv) -> let u = uv[1]
```

make u and v linear

```julia
                nbatch = size(u)[end]
                u = reshape(u, grid.Nu, nbatch)
                v = reshape(v_dw, grid.Nv, nbatch)
```

and concatenate

```julia
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
function upscale_v(grid, resize_v = false)
```

Generate the layer that upscales v
expects as input the linearized concatenation of u and v
returns a tuple (u,v)

```julia
    if !resize_v
        return Chain(uv -> let u = uv[1:(grid.Nu), :], v = uv[(grid.Nu + 1):end, :]
            #println("in upscale")
            #println(typeof(u))
```

reshape u and v on the grid

```julia
            u = reshape(u, grid.nux, grid.nuy, 1, size(u, 2))
            v = reshape(v, grid.nvx, grid.nvy, 1, size(v, 2))
            (u, v)
        end)
    else
        up_v = Chain(
```

extract only the v component

```julia
            uv -> let v = uv[(grid.Nu + 1):end, :]
```

reshape v on the grid

```julia
                v = reshape(v, grid.nvx, grid.nvy, 1, size(v, 2))
                v
            end,
            Upsample(:trilinear, size = (grid.nux, grid.nuy)))
        return Chain(SkipConnection(up_v,
            (v_up, uv) -> let u = uv[1:(grid.Nu), :]
```

make u on the grid

```julia
                u = reshape(u, grid.nux, grid.nuy, 1, size(u, 2))
                (u, v_up)
            end))
    end
end
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

