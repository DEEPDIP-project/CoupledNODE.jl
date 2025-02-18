using JLD2
using CUDA: CUDA
using Lux: Lux

"""
    load_checkpoint(checkfile)

Load a training checkpoint from the specified file.

# Arguments
- `checkfile::String`: The path to the checkpoint file.

# Returns
- `callbackstate`: The state of the callback at the checkpoint.
- `trainstate`: The state of the training process at the checkpoint.
- `epochs_trained::Int`: The number of epochs completed at the checkpoint.

# Example
```julia
callbackstate, trainstate, epochs_trained = load_checkpoint("checkpoint.jld2")
```
"""
function load_checkpoint(checkfile)
    #checkpoint = load_object(checkfile)
    #callbackstate = checkpoint.callbackstate
    #trainstate = checkpoint.trainstate
    (; callbackstate, trainstate) = namedtupleload(checkfile)
    epochs_trained = length(callbackstate.lhist_train)
    @info "Loading checkpoint from $checkfile.\nPrevious training reached epoch $(epochs_trained)."
    return callbackstate, trainstate, epochs_trained
end
