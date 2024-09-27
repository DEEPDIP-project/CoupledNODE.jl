module CoupledNODE

using CUDA: CUDA
ArrayType = CUDA.functional() ? CUDA.CuArray : Array

include("utils.jl")
include("train.jl")
include("models/FNO.jl")
include("models/cnn.jl")
include("loss/loss_priori.jl")
include("loss/loss_posteriori.jl")

# Modules for the examples
include("equations/Burgers.jl")
include("equations/NavierStokes_utils.jl")

end # module CoupledNODE
