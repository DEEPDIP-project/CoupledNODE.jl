module CoupledNODE

include("models/cnn.jl")
# TODO FNO can become an independent pkg and it needs to be tested
#include("models/FNO.jl")

include("loss/loss_priori.jl")
include("loss/loss_posteriori.jl")

include("checkpoints.jl")
include("train.jl")

end # module CoupledNODE
