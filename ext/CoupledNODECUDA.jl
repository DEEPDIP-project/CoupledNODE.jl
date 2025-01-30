module CoupledNODECUDA

using CoupledNODE
using CUDA: CUDA
using Lux: Lux
function ArrayType()
    return CUDA.functional() ? CUDA.CuArray : Array
end
function get_device()
    return CUDA.functional() ? Lux.cpu_device() : Lux.gpu_device()
end

allowscalar = deepcopy(CUDA.allowscalar)

end
