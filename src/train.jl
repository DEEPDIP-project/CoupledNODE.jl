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
        kwargs...)
    dev = cpu ? identity : Lux.gpu_device()
    if !cpu
        ps, st = (ps, st) .|> dev
    end
    @info "Training on" dev
    # Retrieve the callback from kwargs, default to `nothing` if not provided
    callback = get(kwargs, :callback, nothing)
    # Retrieve the training state from kwargs, otherwise create a new one
    tstate = get(kwargs, :tstate, nothing)
    if tstate === nothing
        tstate = Lux.Training.TrainState(model, ps, st, alg)
    end
    loss::Float32 = 0 #NOP TODO: check compatibiity with precision of data
    @info "Lux Training started"
    for epoch in 1:nepochs
        data = train_dataloader()
        _, loss,
        _, tstate = Lux.Training.single_train_step!(
            ad_type, loss_function, data, tstate)
        if callback !== nothing
            callback(tstate.parameters, loss)
        end
    end
    loss, tstate
end
