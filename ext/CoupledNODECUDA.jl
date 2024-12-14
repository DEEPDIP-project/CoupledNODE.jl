module CoupledNODECUDA

using CUDA: CUDA
ArrayType = CUDA.functional() ? CUDA.CuArray : Array

end
