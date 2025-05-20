module NavierStokes
using IncompressibleNavierStokes
using Lux: Lux, relu
using Random: shuffle
using NeuralClosure
using CUDA: CUDA
using Adapt: adapt

include("callback.jl")
include("utils.jl")
include("io.jl")
include("create_data.jl")

end
