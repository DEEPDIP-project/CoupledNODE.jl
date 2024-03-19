function create_nn_cd(nx, ny, nh)
    return Chain(Dense(nx * ny, nh, tanh),
        Dense(nh, nh, tanh),
        Dense(nh, nx * ny))
end
