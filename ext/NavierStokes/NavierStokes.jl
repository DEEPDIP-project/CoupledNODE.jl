module NavierStokes
using IncompressibleNavierStokes
using Lux: Lux, relu
using Random: shuffle
using NeuralClosure
using CUDA: CUDA

include("callback.jl")
include("utils.jl")
include("io.jl")

end
