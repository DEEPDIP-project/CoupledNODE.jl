module fno
using Lux: Lux, relu
using CUDA: CUDA
using Adapt: adapt
using NeuralOperators
using ComponentArrays: ComponentArray

include("model.jl")

export fno_closure

end
