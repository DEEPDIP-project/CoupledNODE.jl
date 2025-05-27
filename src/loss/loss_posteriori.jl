using Zygote: Zygote
using CUDA: CUDA
using Random: shuffle
using LinearAlgebra: norm
using DifferentialEquations: ODEProblem, solve, Tsit5, RK4, remake, EnsembleProblem,
                             EnsembleThreads
using DiffEqGPU: EnsembleGPUArray
using Lux: Lux
using ChainRulesCore: ignore_derivatives

"""
    create_loss_post_lux(rhs; sciml_solver = RK4(), kwargs...)

Creates a loss function for a-posteriori fitting using the given right-hand side (RHS) function `rhs`.
The loss function computes the sum of squared differences between the predicted values and the actual data,
normalized by the sum of squared actual data values.

# Arguments
- `rhs::Function`: The right-hand side function for the ODE problem.
- `sciml_solver::AbstractODEAlgorithm`: (Optional) The SciML solver to use for solving the ODE problem. Defaults to `RK4()`.
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
        force_cpu = false, sciml_solver = RK4(), kwargs...)
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
            prob, sciml_solver; u0 = x, p = ps,
            adaptive = false, dt = dt, save_start = false, saveat = saveat_times, kwargs...)
        if size(pred)[4] != size(uref)[4]
            @warn "Instability in the loss function. The predicted and target data have different sizes."
            @info "Predicted size: $(size(pred))"
            @info "Target size: $(size(uref))"
            return Inf, st, (; y_pred = pred)
        end

        loss = sum(
            sum((pred[inside..., :, :] .- uref) .^ 2, dims = (1, 2, 3)) ./
            sum(abs2, uref, dims = (1, 2, 3))
        ) / (nts-1)
        return loss, st, (; y_pred = nothing)
    end

    if force_cpu || true
        ensemble_type = EnsembleThreads()
    else
        ensemble_type = CUDA.functional() ? EnsembleGPUArray(CUDA.CUDABackend()) :
                        EnsembleThreads()
        # unfortunately, EnsembleGPUKernel does not work for complicated rhs
    end

    function _loss_ensemble(model, ps, st, (all_u, all_t))
        @error "Ensemble loss is broken. Follow this issue for updates: https://github.com/DEEPDIP-project/CoupledNODE.jl/issues/207"
        nsamp = size(all_u, ndims(all_u) - 1)
        nts = size(all_u, ndims(all_u))

        t_indices = 2:nts
        saveat_times = Array(all_t[1, t_indices]) # This has to be Array even for GPU solvers

        x0, tspan, uref, pred = nothing, nothing, nothing, nothing

        # Define how each problem varies
        function prob_func(prob, i, repeat)
            CUDA.allowscalar() do
                xᵢ = all_u[griddims..., :, i, 1]
                tᵢ = all_t[i, :]
                remake(prob; u0 = xᵢ, tspan = (tᵢ[1], tᵢ[end]))
            end
        end

        CUDA.allowscalar() do
            x0 = all_u[griddims..., :, 1, 1]
            tspan = (all_t[1, 1], all_t[1, end])
        end
        base_prob = ODEProblem(rhs, x0, tspan, ps)

        ensemble_prob = EnsembleProblem(
            base_prob, prob_func = prob_func, safetycopy = false)

        sols = solve(
            ensemble_prob, sciml_solver, ensemble_type;
            dt = dt,
            trajectories = nsamp,
            saveat = saveat_times,
            adaptive = false,
            save_start = false
        )

        # Compute loss
        losses = map(1:nsamp) do i
            CUDA.allowscalar() do
                uref = all_u[inside..., :, i, t_indices]
                pred = ArrayType()(sols.u[i][inside..., :, :]) # It is FUNDAMENTAL to convert to ArrayType
            end

            if size(pred) != size(uref)
                @warn "Shape mismatch in sample $i: $(size(pred)) vs $(size(uref))"
                return Inf, st, (; y_pred = sols[i])
            end

            sum(
                sum((pred .- uref) .^ 2, dims = (1, 2, 3)) ./
                sum(abs2, uref, dims = (1, 2, 3))
            )
        end
        return sum(losses) / (nsamp*(nts - 1)), st, (; y_pred = nothing)
    end

    if !ensemble
        return _loss_function
    else
        @info "Using ensemble loss function"
        return _loss_ensemble
    end
end
