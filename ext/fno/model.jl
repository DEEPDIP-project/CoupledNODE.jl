function fno_closure(;
        T = Float32,
        chs = (2, 64, 64, 64, 2),
        activation = Lux.gelu,
        modes = (4, 4),
        use_cuda = false,
        rng
)
    if use_cuda
        dev = x->adapt(CuArray, x)
    else
        dev = Lux.cpu_device()
    end

    @warn "*** FNO is using the following device: $(dev) "

    chain = FourierNeuralOperator(
        activation;
        chs = chs,
        modes = modes,
        permuted = Val(true)
    )
    params, state = Lux.setup(rng, chain)
    state = state |> dev
    params = params |> dev
    # [!BUG] FourierNeuralOperator does not support ComponentArrays, so it can not be trained a posteriori
    #params = ComponentArray(params)# |> dev
    (chain, params, state)
end

function load_seeds(conf)
    data = conf["seeds"]
    seeds = (;
        dns = data["dns"],
        θ_start = data["θ_start"],
        prior = data["prior"],
        post = data["post"]
    )
    return seeds
end

function load_fno_params(conf::Dict)
    closure_type = conf["closure"]["type"]
    if closure_type != "fno"
        @error "Model type $closure_type not supported by this function"
        return
    end

    T = eval(Meta.parse(conf["T"]))

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

    # Construct the fno call
    data = conf["closure"]
    seeds = load_seeds(conf)

    closure, θ_start,
    st = fno_closure(
        T = T,
        chs = Tuple(data["channels"]),
        #activation = map(eval_field, data["activation"]),
        modes = Tuple(data["modes"]),
        use_cuda = CUDA.functional() ? true : false,
        rng = eval_field(data["rng"], seeds)
    )

    return closure, θ_start, st
end
