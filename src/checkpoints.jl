
using JLD2

function load_checkpoint(checkfile)
    checkpoint = load_object(checkfile)
    callbackstate = checkpoint.callbackstate
    trainstate = checkpoint.trainstate
    epochs_trained = length(callbackstate.lhist_train)
    @info "Loading checkpoint from $checkfile.\nPrevious training reached epoch $(epochs_trained)."
    return callbackstate, trainstate, epochs_trained
end
