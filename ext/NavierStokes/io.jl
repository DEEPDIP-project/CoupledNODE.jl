using Random
using YAML
using CoupledNODE

"""
    read_config(filename::String)

Reads a YAML configuration file and returns the parsed content as a dictionary.

# Arguments
- `filename::String`: The path to the YAML configuration file.

# Returns
- `conf::Dict`: The parsed configuration dictionary.
"""
function read_config(filename)
    conf = YAML.load_file(filename)
    return conf
end

"""
    load_params(conf::Dict)

Loads and evaluates the parameters from the configuration dictionary.

# Arguments
- `conf::Dict`: The configuration dictionary.

# Returns
- `params::NamedTuple`: A named tuple containing the evaluated parameters.
"""
function load_params(conf)
    data = conf["params"]
    T = eval(Meta.parse(conf["T"]))
    function eval_field(field, T)
        if field isa String
            field = "T=$T; $field"
            return eval(Meta.parse(field))
        else
            return field
        end
    end

    params = (;
        D = data["D"],
        lims = (T(data["lims"][1]), T(data["lims"][2])),
        Re = T(data["Re"]),
        tburn = T(data["tburn"]),
        tsim = T(data["tsim"]),
        savefreq = data["savefreq"],
        ndns = data["ndns"],
        nles = data["nles"],
        filters = tuple(map(f -> eval_field(f, T), data["filters"])...),
        icfunc = eval_field(data["icfunc"], T),
        method = eval_field(data["method"], T),
        bodyforce = eval_field(data["bodyforce"], T),
        processors = eval_field(data["processors"], T),
        issteadybodyforce = data["issteadybodyforce"],
        Δt = T(data["Δt"]),
        backend = data["backend"]
    )

    return params
end

"""
    load_seeds(conf::Dict)

Loads the seed values from the configuration dictionary.

# Arguments
- `conf::Dict`: The configuration dictionary.

# Returns
- `seeds::NamedTuple`: A named tuple containing the seed values.
"""
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

"""
    load_model(conf::Dict)

Loads the model parameters based on the model type specified in the configuration dictionary.

# Arguments
- `conf::Dict`: The configuration dictionary.

# Returns
- `model::Any`: The loaded model parameters.

# Throws
- `Error`: If the model type is not supported.
"""
function load_model(conf)
    model_type = conf["closure"]["type"]
    if model_type == "cnn"
        return load_cnn_params(conf)
    else
        error("Model type not supported")
    end
end

"""
    load_cnn_params(conf::Dict)

Loads the parameters for a CNN model from the configuration dictionary.

# Arguments
- `conf::Dict`: The configuration dictionary.

# Returns
- `closure::Any`: The CNN closure.
- `θ_start::Any`: The initial parameters for the CNN.
- `st::Any`: The state of the CNN.
"""
function load_cnn_params(conf)
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
    closure, θ_start, st = CoupledNODE.cnn(
        T = T,
        D = D,
        data_ch = D,
        radii = data["radii"],
        channels = data["channels"],
        activations = map(eval_field, data["activations"]),
        use_bias = data["use_bias"],
        rng = eval_field(data["rng"], seeds)
    )

    return closure, θ_start, st
end
