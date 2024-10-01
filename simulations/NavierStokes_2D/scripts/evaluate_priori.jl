using JLD2: @load

T = Float32
ArrayType = Array
rng = Random.Xoshiro(123)
ig = 1 # index of the LES grid to use.

# Create model
using CoupledNODE: cnn
closure, _, _ = cnn(;
    setup = setups[ig],
    radii = [3, 3],
    channels = [2, 2],
    activations = [tanh, identity],
    use_bias = [false, false],
    rng
)

# Load model params
outdir = "simulations/NavierStokes_2D/outputs"
@load "$outdir/trained_model_priori.jld2" θ_priori st
