using SciMLSensitivity
import Lux
import Juno
import Optimization, OptimizationOptimisers
"""
    optimize(θ, loss, ad_type=Optimization.AutoZygote(), alg::AbstractOptimizationAlgorithm=OptimizationOptimisers.Adam(0.1), args...; kwargs...)

Optimizes the given parameters `θ` with a loss function `loss` using the optimization algorithm `alg`.
Check the documentation of `Optimization.solve` for more details and optional arguments.

## Arguments
- `θ`: The initial parameter values.
- `loss`: The loss function to be minimized.
- `ad_type`: The automatic differentiation type to be used. Default is `Optimization.AutoZygote()`.
- `alg`: The optimization algorithm to be used. Default is `OptimizationOptimisers.Adam(0.1)`.
- `args...`: Additional positional arguments to be passed to the optimization algorithm.
- `kwargs...`: Additional keyword arguments to be passed to the optimization algorithm.

## Returns
The optimized parameters (θ).
"""
function optimize(θ, loss, ad_type = Optimization.AutoZygote(),
        alg = OptimizationOptimisers.Adam(0.1),
        args...; kwargs...)
    optf = Optimization.OptimizationFunction((x, p) -> loss(x), ad_type)
    optprob = Optimization.OptimizationProblem(optf, θ)
    Optimization.solve(
        optprob,
        alg,
        args...;
        kwargs...
    ).u
end

function train(dataloaders,
        loss,
        θ;
        niter = 100,
        ncallback = 1,
        callback = (state, i, θ) -> println("Iteration $i of $niter"),
        callbackstate = nothing
)
    θ = optimize(θ, loss;
        maxiters = niter, progress = true, callback = callback)
    (; optstate, θ, callbackstate)
end

function train(model, ps, st, train_dataloader, loss_function;
        nepochs = 100, ad_type = Optimization.AutoZygote(),
        alg = OptimizationOptimisers.Adam(0.1), cpu::Bool = false, kwargs...)
    dev = cpu ? Lux.cpu_device() : Lux.gpu_device()
    # Retrieve the callback from kwargs, default to `nothing` if not provided
    callback = get(kwargs, :callback, nothing)
    tstate = Lux.Training.TrainState(model, ps, st, alg)
    loss::Float32 = 0 #NOP
    Juno.@progress for epoch in 1:nepochs
        data = train_dataloader()
        _, loss, _, tstate = Lux.Training.single_train_step!(
            ad_type, loss_function, data, tstate)
        if callback !== nothing
            callback(tstate.parameters, loss)
        end
    end
    loss, tstate
end
