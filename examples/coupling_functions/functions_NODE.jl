
# I define the NeuralODE using ResNet skip blocks to add the closure
function create_f_NODE(NN, f_u; is_closed=false)
    return Chain(
        SkipConnection(NN, (f_NN, u) -> is_closed ? f_NN + f_u(u) : f_u(u)),
    )
end

# NeuralODE representing the experimental observation
function create_NODE_obs()
    return Chain( 
        u -> f_o(u),
    )
end

function create_f_CNODE(F_u, G_v, grid; is_closed=false)
    return Chain(
        # from the input which is the concatenation of u and v, apply F to the first half and G to the second half
        uv -> let u = uv[1:grid.nux,1:grid.nuy], v = uv[grid.nux+1:end,grid.nuy+1:end]
            BlockDiagonal([F_u(u, v, grid), G_v(u, v, grid)])
        end
        # no closure yet
    )
    
end


# This object contains the grid information
struct Grid
    dux::Float64
    duy::Float64
    nux::Int
    nuy::Int
    dvx::Float64
    dvy::Float64
    nvx::Int
    nvy::Int
end