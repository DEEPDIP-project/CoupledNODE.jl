module CNO

using ConvolutionalNeuralOperators: create_CNO
using ComponentArrays: ComponentArray
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

    # Model configuration
    data = conf["closure"]
    NS = Base.get_extension(CoupledNODE, :NavierStokes)
    seeds = NS.load_seeds(conf)
    use_cuda = CUDA.functional() ? true : false
    if use_cuda
        dev = Lux.gpu_device()
    else
        dev = Lux.cpu_device()
    end
    closure = create_CNO(
        T = T,
        N = data["size"],
        D = D,
        cutoff = data["cutoff"],
        ch_sizes = data["channels"],
        activations = map(eval_field, data["activations"]),
        down_factors = data["down_factors"],
        k_radii = data["radii"],
        bottleneck_depths = haskey(data, "bottleneck_depths") ? data["bottleneck_depths"] :
                            nothing
    )
    params, state = Lux.setup(eval_field(data["rng"], seeds), closure)
    st = state |> dev
    θ_start = ComponentArray(params) |> dev

    return closure, θ_start, st
end

export load_cno_params

end
