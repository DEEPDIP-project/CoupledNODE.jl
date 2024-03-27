module CoupledNODE

import CUDA
ArrayType = CUDA.functional() ? CuArray : Array

include("loss.jl")
include("grid.jl")
include("NODE.jl")
include("derivatives.jl")
include("utils.jl")

end # module CoupledNODE
