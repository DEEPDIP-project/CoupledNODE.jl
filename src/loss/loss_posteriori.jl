using Zygote: Zygote
using CUDA: CUDA
using Random: shuffle
using LinearAlgebra: norm
using DifferentialEquations: ODEProblem, solve, Tsit5, RK4
using Lux: Lux
using ChainRulesCore: ignore_derivatives

"""
    create_loss_post_lux(rhs; sciml_solver = RK4(), kwargs...)

Creates a loss function for a-posteriori fitting using the given right-hand side (RHS) function `rhs`.
The loss function computes the sum of squared differences between the predicted values and the actual data,
normalized by the sum of squared actual data values.

# Arguments
- `rhs::Function`: The right-hand side function for the ODE problem.
- `sciml_solver::AbstractODEAlgorithm`: (Optional) The SciML solver to use for solving the ODE problem. Defaults to `Tsit5()`.
- `kwargs...`: Additional keyword arguments to pass to the solver.

# Returns
- `loss_function::Function`: A function that computes the loss given a model, parameters, state, and data `(u, t)`.

## `loss_function` Arguments
- `model`: The model to evaluate.
- `ps`: Parameters for the model.
- `st`: State of the model.
- `(u, t)`: Tuple containing the data `u` and time points `t`.

## `loss_function` Returns
- `loss`: The computed loss value.
- `st`: The model state (unchanged).
- `metadata::NamedTuple`: A named tuple containing the predicted values `y_pred`.
This makes it compatible with the Lux ecosystem.
"""
function create_loss_post_lux(
        rhs, griddims, inside, dt; ensemble = false,
        force_cpu = false, sciml_solver = Tsit5(), kwargs...)
    function ArrayType()
        if force_cpu
            return Array
        end
        return CUDA.functional() ? CUDA.CuArray : Array
    end

    function _loss_function(model, ps, st, (all_u, all_t))
        nsamp = size(all_u, ndims(all_u) - 1)
        @assert nsamp == 1 "Single-sample loss function is not supported in this context. Use ensemble loss function instead."
        nts = size(all_u, ndims(all_u))

        uref, x, saveat_times, tspan,
        prob, pred = nothing, nothing, nothing, nothing, nothing, nothing

        CUDA.allowscalar() do
            uref = all_u[inside..., :, 1, 2:nts]
            x = all_u[griddims..., :, 1, 1]
            saveat_times = Array(all_t[1, 2:nts])
            tspan = (all_t[1, 1], all_t[1, end])
        end
        prob = ODEProblem(rhs, x, tspan, ps)
        pred = solve(
            prob,
            sciml_solver;
            u0 = x,
            p = ps,
            tspan = tspan,
            adaptive = true,
            #dt = dt,
            save_start = false,
            saveat = saveat_times,
            tstops = saveat_times,
            kwargs...
        )

        if pred.retcode != :Success
            @warn "ODE solver did not converge. Retcode: $(pred.retcode)"
            return 100000, st, (; y_pred = nothing)
        end
        if size(pred)[4] != size(uref)[4]
            @warn "Instability in the loss function. The predicted and target data have different sizes."
            @info "Predicted size: $(size(pred))"
            @info "Target size: $(size(uref))"
            @info "Pred time points: $(pred.t)"
            @info "Target time points: $(all_t[1, 2:nts])"
            return Inf, st, (; y_pred = nothing)
        end

        loss = Lux.MSELoss()(pred[inside..., :, :], uref)

        if isnan(loss) || isinf(loss)
            @warn "Loss is NaN or Inf. Returning Inf."
            @info "max(pred - uref): $(maximum(pred[inside..., :, :] .- uref))"
            return Inf, st, (; y_pred = nothing)
        end
        if isapprox(loss, 0.0)
            @warn "Loss is approximately zero. I ignore this point cause it might be a numerical issue."
            return Inf, st, (; y_pred = nothing)
        end

        return loss, st, (; y_pred = nothing)
    end

    function _loss_ensemble(model, ps, st, (all_u, all_t))
        nsamp = size(all_u, ndims(all_u) - 1)
        nts = size(all_u, ndims(all_u))

        t_indices = 2:nts
        saveat_times = Array(all_t[1, t_indices]) # This has to be Array even for GPU solvers

        xi, tspan, uref, pred = nothing, nothing, nothing, nothing, nothing

        ignore_derivatives() do
            CUDA.allowscalar() do
                tspan = (all_t[1, 1], all_t[1, end]) # tspan is the same for all problems
                xi = [all_u[griddims..., :, i, 1] for i in 1:nsamp]
            end
        end

        prob = ODEProblem(rhs, nothing, tspan, nothing)

        sols = [solve(
                    prob,
                    sciml_solver;
                    u0 = xi[i],
                    p = ps,
                    tspan = tspan,
                    #dt = dt,
                    saveat = saveat_times,
                    tstops = saveat_times,
                    adaptive = true,
                    save_start = false,
                    kwargs...
                )[inside..., :, :]
                for i in 1:nsamp]

        loss = Lux.MSELoss()(stack(sols, dims = 4), all_u[inside..., :, :, t_indices])
        return loss, st, (; y_pred = nothing)
    end

    if !ensemble
        return _loss_function
    else
        @info "Using ensemble loss function"
        return _loss_ensemble
    end
end
