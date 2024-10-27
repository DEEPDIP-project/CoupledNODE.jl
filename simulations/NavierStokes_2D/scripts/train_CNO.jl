using CoupledNODE: cnn, train, create_loss_post_lux, create_callback
using CoupledNODE.NavierStokes: create_right_hand_side_with_closure
using DifferentialEquations: ODEProblem, solve, Tsit5
using IncompressibleNavierStokes: IncompressibleNavierStokes as INS
using JLD2: @save
using Optimization: Optimization
using OptimizationOptimisers: OptimizationOptimisers
using Random: Random

T = Float32
rng = Random.Xoshiro(123)
ig = 1 # index of the LES grid to use.
include("preprocess_posteriori.jl")

using ComponentArrays: ComponentArray
using Lux: Lux
u = io_post[ig].u[:, :, :, 1, 1:50]
T = setups[1].T
d = D = setups[1].grid.dimension()
u0 = u

#************88
# Test that the downsampler and upsampler work
using TestImages: testimage
using Plots: heatmap, plot, plot!


N0 = 512
u0 = zeros(T, N0, N0, D, 6)
u0[:, :, 1, 1] = testimage("cameraman")
u0[:, :, 2, 1] = testimage("cameraman")
u0[:, :, 1, 2] = testimage("brick_wall_512")
u0[:, :, 2, 2] = testimage("brick_wall_512")
u0[:, :, 1, 3] = testimage("fabio_gray_512")
u0[:, :, 2, 3] = testimage("fabio_gray_512")
u0[:, :, 1, 4] = testimage("lena_gray_512")
u0[:, :, 2, 4] = testimage("lena_gray_512")
u0[:, :, 1, 5] = testimage("livingroom")
u0[:, :, 2, 5] = testimage("livingroom")
u0[:, :, 1, 6] = testimage("pirate")
u0[:, :, 2, 6] = testimage("pirate")
typeof(u0)
cutoff = 0.5

# downsize the input which would be too large to autodiff
using CoupledNODE: create_CNOdownsampler
down_factor = 2
ds = create_CNOdownsampler(T, D, N0, down_factor, cutoff)
u = ds(u0)
N = size(u, 1)
heatmap(u[:, :, 2, 3], aspect_ratio = 1, title = "downsampled")

using CoupledNODE: create_CNO
ch_ = [1,1]
df = [2,2]
k_rad = [1,1]
bd = [2,2,2]
model = create_CNO(T=T, N=N, D=D, cutoff=cutoff, ch_sizes=ch_, down_factors=df, k_radii=k_rad, bottleneck_depths = bd);
θ, st = Lux.setup(rng, model);
using ComponentArrays: ComponentArray
θ = ComponentArray(θ)
size(model(u, θ, st)[1])
size(u)
heatmap(model(u, θ, st)[1][:, :, 1, 4], aspect_ratio = 1, title = "model(u0)")


using Zygote: Zygote
model(u, θ, st)[1] .- u
function loss(θ)
    û = model(u, θ, st)[1]
    return sum(abs2,(û .- u)./ u)
end
loss(θ)
g = Zygote.gradient(θ->loss(θ), θ)

function callback(p, l_train)
    @info "Training Loss: $(l_train)"
    println(p.u[1])
    false
end

# test training with optimize
using Optimization: Optimization
optf = Optimization.OptimizationFunction(
    (p, _) -> loss(p),
    Optimization.AutoZygote()
)
optprob = Optimization.OptimizationProblem(optf, θ)
optim_result, optim_t, optim_mem, _ = @timed Optimization.solve(
    optprob,
    OptimizationOptimisers.Adam(0.2);
    maxiters = 200,
    callback = callback,
    progress = true
)
θ_p = optim_result.u
heatmap(model(u, θ_p, st)[1][:, :, 1, 3], aspect_ratio = 1, title = "model(u0)")
θ = θ_p

