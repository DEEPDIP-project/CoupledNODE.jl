module CoupledNODE

using CUDA: CUDA
ArrayType = CUDA.functional() ? CUDA.CuArray : Array

include("models/FNO.jl")
include("models/cnn.jl")

include("loss/loss_priori.jl")
include("loss/loss_posteriori.jl")

include("equations/NavierStokes_utils.jl")

include("utils.jl")
include("train.jl")

end # module CoupledNODE
