using SciMLSensitivity
using Lux: Lux
using Zygote: Zygote
using Optimization: Optimization
using OptimizationOptimisers: OptimizationOptimisers
using CairoMakie: save

function train(model, ps, st, train_dataloader, loss_function;
        nepochs = 100,
        ad_type = Optimization.AutoZygote(),
        alg = OptimizationOptimisers.Adam(0.1),
        cpu::Bool = false,
        kwargs...)
    dev = cpu ? Lux.cpu_device() : Lux.gpu_device()
    ps, st = (ps, st) .|> dev
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
        data = Zygote.@ignore dev(train_dataloader())
        _, loss, _, tstate = Lux.Training.single_train_step!(
            ad_type, loss_function, data, tstate)
        if callback !== nothing
            callback(tstate.parameters, loss)
        end
    end
    loss, tstate
end
