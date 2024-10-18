module CoupledNODE

using CUDA: CUDA
ArrayType = CUDA.functional() ? CUDA.CuArray : Array

include("utils.jl")
include("train.jl")

include("models/cnn.jl")
include("models/FNO.jl")
include("models/transformer.jl")
include("models/CNO_utils.jl")
include("models/CNO.jl")

include("loss/loss_priori.jl")
include("loss/loss_posteriori.jl")

include("equations/NavierStokes_utils.jl")

end # module CoupledNODE
