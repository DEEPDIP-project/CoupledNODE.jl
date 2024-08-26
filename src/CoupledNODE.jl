module CoupledNODE

import CUDA
ArrayType = CUDA.functional() ? CUDA.CuArray : Array

include("NODE.jl")
#include("derivatives.jl")
include("utils.jl")
include("models/FNO.jl")
include("loss/loss_priori.jl")
include("loss/loss_posteriori.jl")

# Modules for the examples
include("equations/Burgers.jl")
include("equations/NavierStokes.jl")

end # module CoupledNODE
