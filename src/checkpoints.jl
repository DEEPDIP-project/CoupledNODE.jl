using JLD2
using CUDA: CUDA
using Lux: Lux

"""
    namedtupleload(file)

Load a JLD2 file and convert it to a named tuple.

# Arguments
- `file::String`: The path to the JLD2 file.

# Returns
- `NamedTuple`: The contents of the file as a named tuple.
"""
function namedtupleload(file)
    dict = load(file)
    k, v = keys(dict), values(dict)
    pairs = @. Symbol(k) => v
    (; pairs...)
end

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
    (; callbackstate, trainstate) = namedtupleload(checkfile)
    epochs_trained = length(callbackstate.lhist_train)
    @info "Loading checkpoint from $checkfile.\nPrevious training reached epoch $(epochs_trained)."
    return callbackstate, trainstate, epochs_trained
end

"""
    save_checkpoint(checkfile, callbackstate, trainstate)

Save the current training state to a checkpoint file.

# Arguments
- `checkfile::String`: The path to the checkpoint file.
- `callbackstate`: The current state of the callback.
- `trainstate`: The current state of the training process.
"""
function save_checkpoint(checkfile, callbackstate, trainstate)
    @info "Saving checkpoint to $checkfile."
    c = callbackstate |> cpu_device()
    t = trainstate |> cpu_device()
    jldsave(checkfile; callbackstate = c, trainstate = t)
end
