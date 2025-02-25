module AttentionCNN

using AttentionLayer: attentioncnn

function load_cnn_params(conf)
    closure_type = conf["closure"]["type"]
    if closure_type != "attentioncnn"
        @error "Model type $closure_type not supported by this function"
        return
    end

    T = eval(Meta.parse(conf["T"]))
    D = conf["params"]["D"]

    # Evaluate activations and rng
    function eval_field(field, s = nothing)
        if field isa String
            if s != nothing
                field = "seeds=$s; $field"
            end
            return eval(Meta.parse(field))
        else
            return field
        end
    end

    # Construct the cnn call
    data = conf["closure"]
    seeds = load_seeds(conf)
    closure, θ_start, st = attentioncnn(
        T = T,
        D = D,
        data_ch = D,
        radii = data["radii"],
        channels = data["channels"],
        activations = map(eval_field, data["activations"]),
        use_bias = data["use_bias"],
        use_attention = data["use_attention"],
        emb_sizes = data["emb_sizes"],
        patch_sizes = data["patch_sizes"],
        n_heads = data["n_heads"],
        sum_attention = data["sum_attention"],
        rng = eval_field(data["rng"], seeds),
        use_cuda = CUDA.functional() ? true : false
    )

    return closure, θ_start, st
end

end
