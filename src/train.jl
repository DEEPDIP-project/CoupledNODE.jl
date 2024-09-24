using SciMLSensitivity
using Lux: Lux
using Juno: Juno
using Zygote: Zygote
using Optimization: Optimization
using OptimizationOptimisers: OptimizationOptimisers

function train(model, ps, st, train_dataloader, loss_function;
        nepochs = 100, ad_type = Optimization.AutoZygote(),
        alg = OptimizationOptimisers.Adam(0.1), cpu::Bool = false, kwargs...)
    dev = cpu ? Lux.cpu_device() : Lux.gpu_device()
    # Retrieve the callback from kwargs, default to `nothing` if not provided
    callback = get(kwargs, :callback, nothing)
    tstate = Lux.Training.TrainState(model, ps, st, alg)
    loss::Float32 = 0 #NOP
    Juno.@progress for epoch in 1:nepochs
        data = Zygote.@ignore train_dataloader()
        _, loss, _, tstate = Lux.Training.single_train_step!(
            ad_type, loss_function, data, tstate)
        if callback !== nothing
            callback(tstate.parameters, loss)
        end
    end
    loss, tstate
end
