using Test
using CoupledNODE: save_checkpoint, load_checkpoint
using JLD2
using Random
using Adapt

@testset "Checkpoint Tests" begin
    # Create dummy callbackstate and trainstate
    callbackstate = (lhist_train = [1, 2, 3], lhist_val = [1, 2, 3])
    trainstate = (weights = rand(3, 3), biases = rand(3))

    # Define checkpoint file path
    checkfile = "test_checkpoint.jld2"

    # Save checkpoint
    save_checkpoint(checkfile, callbackstate, trainstate)

    # Load checkpoint
    loaded_callbackstate, loaded_trainstate, epochs_trained = load_checkpoint(checkfile)

    # Test loaded values
    @test loaded_callbackstate == callbackstate
    @test loaded_trainstate == trainstate
    @test epochs_trained == length(callbackstate.lhist_train)

    # Clean up
    rm(checkfile, force=true)
end
