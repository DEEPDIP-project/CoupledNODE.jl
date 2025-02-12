module CoupledNODECUDA_ext

# TODO: this extension is basically pointless at the moment and it causes a lot of trouble

using CoupledNODE
using CUDA: CUDA
using CUDSS
using Lux: Lux
using LuxCUDA
function ArrayType()
    return CUDA.functional() ? CUDA.CuArray : Array
end
function get_device()
    return CUDA.functional() ? Lux.gpu_device() : Lux.cpu_device()
end

allowscalar = deepcopy(CUDA.allowscalar)

end
