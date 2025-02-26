module CNO

using ConvolutionalNeuralOperators
using Lux: Lux, relu
using Random
using CoupledNODE
using CUDA: CUDA

function load_cno_params(conf)
    closure_type = conf["closure"]["type"]
    if closure_type != "cno"
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
    NS = Base.get_extension(CoupledNODE, :NavierStokes)
    seeds = NS.load_seeds(conf)
    closure, θ_start, st = attentioncnn(
        T = T,
        D = D,
        N = data["N"],
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

    # Model configuration
    data = conf["closure"]
    NS = Base.get_extension(CoupledNODE, :NavierStokes)
    seeds = NS.load_seeds(conf)
    model = create_CNO(
        T = T,
        N = N,
        D = D,
        cutoff = data["cutoff"],
        ch_sizes = data["channels"],
        activations = map(eval_field, data["activations"]),
        down_factors = data["down_factors"],
        k_radii = data["radii"],
        bottleneck_depths = data["bottleneck_depths"]
    )
    rng = eval_field(data["rng"], seeds)
    θ, st = Lux.setup(rng, model)
    θ = ComponentArray(θ)

    return closure, θ_start, st
end

export load_cno_params

end
