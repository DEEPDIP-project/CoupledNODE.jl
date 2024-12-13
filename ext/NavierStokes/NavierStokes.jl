module NavierStokes
using IncompressibleNavierStokes
using Lux: Lux
using Random: shuffle
#using Pkg
#Pkg.add(url = "https://github.com/DEEPDIP-project/NeuralClosure.jl.git")
using NeuralClosure
include("callback.jl")
include("utils.jl")
include("io.jl")

#export create_io_arrays_posteriori, create_dataloader_posteriori, create_right_hand_side_with_closure
#export create_io_arrays_priori, create_dataloader_prior
#export read_config, load_cnn_params, load_params, load_seeds, load_model
#export create_callback

end
