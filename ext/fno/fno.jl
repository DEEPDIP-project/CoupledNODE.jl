module fno
using Lux: Lux, relu
using CUDA
using Adapt: adapt
using NeuralOperators
using ComponentArrays: ComponentArray
using Random

include("model.jl")

export fno_closure

end
