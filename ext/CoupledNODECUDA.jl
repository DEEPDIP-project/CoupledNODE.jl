module CoupledNODECUDA

using CoupledNODE
using CUDA: CUDA
using CUDSS
using Lux: Lux
using LuxCUDA
function ArrayType()
    return CUDA.functional() ? CUDA.CuArray : Array
end
function get_device()
    return CUDA.functional() ? Lux.cpu_device() : Lux.gpu_device()
end

allowscalar = deepcopy(CUDA.allowscalar)

# GPU version of interpolate without circshift
function gpu_interpolate(A, D, dir)
    @warn "Using GPU version of interpolate"
    (i, a) = A
    if i > D
        return a  # Nothing to interpolate for extra layers
    end

    shift_amount = dir
    shifted = similar(a)  # Create an array of the same size as `a`

    if shift_amount > 0
        shifted[shift_amount+1:end, :] .= a[1:end-shift_amount, :]
    elseif shift_amount < 0
        shifted[1:end+shift_amount, :] .= a[1-shift_amount:end, :]
    else
        shifted .= a
    end

    staggered = a .+ shifted
    staggered ./ 2
end

end
