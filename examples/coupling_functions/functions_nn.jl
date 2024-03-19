
function create_dumb_nn(grid)
    ninput = grid.nux * grid.nuy + grid.nux * grid.nuy
    noutput = grid.nux * grid.nuy
    return Chain(uv -> reshape(uv, grid.nux * grid.nuy * 2, :),
        Dense(ninput,
            noutput,
            leakyrelu,
            init_weight = Lux.zeros32,
            init_bias = Lux.zeros32),
        ReshapeLayer((grid.nux, grid.nuy)))
end
function create_cnn(grid)
    r_cnn = [1, 1, 2, 2]
    ch_cnn = [2, 4, 8, 16, 36]
    σ = [leakyrelu, leakyrelu, leakyrelu, leakyrelu]
    pool_k = [2, 2, 2, 2]
    pad = r_cnn
    stride = [1, 1, 1, 1]
    # the output of the cnn will have this size
    nout_cnn = grid.nux
    for i in eachindex(r_cnn)
        # output size of the cnn
        nout_cnn = Int(floor((nout_cnn - (2 * r_cnn[i] + 1) + 2 * pad[i]) / stride[i] + 1))
        # this is for the pooling
        nout_cnn = Int(floor((nout_cnn - (pool_k[i])) / pool_k[i] + 1))
    end
    n_features = ch_cnn[end] * nout_cnn * nout_cnn
    # this cnn takes as input a 2d field and returns a 2d field
    return Chain((Chain(
            # Before convolution I do a periodic padding
            uv -> NNlib.pad_circular(uv, r_cnn[i], dims = [1, 2]),
            uv -> reshape(uv, size(uv, 1), size(uv, 2), size(uv, 3), :),
            Conv((2 * r_cnn[i] + 1, 2 * r_cnn[i] + 1),
                ch_cnn[i] => ch_cnn[i + 1],
                σ[i];
                pad = 0,
                stride = stride[i]),
            MaxPool((pool_k[i], pool_k[i]))) for i in eachindex(r_cnn))...,
        FlattenLayer(),
        Dense(n_features, grid.Nu, leakyrelu),
        ReshapeLayer((grid.nux, grid.nuy)))
end
