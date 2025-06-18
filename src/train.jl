using SciMLSensitivity
using Lux: Lux
using Zygote: Zygote
using Optimization: Optimization
using OptimizationOptimisers: OptimizationOptimisers
using ChainRulesCore: ignore_derivatives

function train(model, ps, st, train_dataloader, loss_function;
        nepochs = 50,
        ad_type = Optimization.AutoZygote(),
        alg = OptimizationOptimisers.Adam(0.1),
        cpu::Bool = false,
        λ = nothing,
        kwargs...)
    dev = cpu ? identity : Lux.gpu_device()
    if !cpu
        ps, st = (ps, st) .|> dev
    end
    @info "Training on" dev
    isnothing(λ) || @info "Using weight decay λ = $λ"
    # Retrieve the callback from kwargs, default to `nothing` if not provided
    callback = get(kwargs, :callback, nothing)
    # Retrieve the training state from kwargs, otherwise create a new one
    tstate = get(kwargs, :tstate, nothing)
    if tstate === nothing
        tstate = Lux.Training.TrainState(model, ps, st, alg)
    end
    T = eltype(ps)
    loss = zero(T)
    @info "Lux Training started"
    for epoch in 1:nepochs
        data = train_dataloader()
        grads, loss,
        st, tstate = Lux.Training.compute_gradients(
            ad_type, loss_function, data, tstate)
        isnothing(λ) || @.(grads += λ * tstate.parameters) # Weight decay
        Lux.Training.apply_gradients!(tstate, grads)
        if callback !== nothing
            callback(tstate.parameters, loss)
        end
    end
    loss, tstate
end
