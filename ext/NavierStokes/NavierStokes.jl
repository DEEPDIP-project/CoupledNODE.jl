module NavierStokes
using IncompressibleNavierStokes
using Lux: Lux
using Random: shuffle
using NeuralClosure

include("callback.jl")
include("utils.jl")
include("io.jl")

end
