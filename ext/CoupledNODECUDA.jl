module CoupledNODECUDA

using CoupledNODE
using CUDA: CUDA
function ArrayType()
    return CUDA.functional() ? CUDA.CuArray : Array
end

allowscalar = deepcopy(CUDA.allowscalar)

end
