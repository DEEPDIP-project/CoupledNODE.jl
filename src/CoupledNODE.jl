module CoupledNODE

import CUDA
ArrayType = CUDA.functional() ? CUDA.CuArray : Array

#include("loss_priori.jl")
#include("loss_posteriori.jl")
#include("grid.jl")
#include("NODE.jl")
#include("FNO.jl")
#include("derivatives.jl")
#include("utils.jl")
#
## Modules for the examples
#include("Burgers.jl")

end # module CoupledNODE
