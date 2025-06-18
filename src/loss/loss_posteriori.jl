using Zygote: Zygote
using CUDA: CUDA
using Random: shuffle
using LinearAlgebra: norm
using DifferentialEquations: ODEProblem, solve, Tsit5, RK4
using DiffEqFlux: group_ranges
using Lux: Lux
using ChainRulesCore: ignore_derivatives
using SciMLBase

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

function mymultiple_shoot(p, ode_data, tsteps, prob::ODEProblem, loss_function::F,
        continuity_loss::C, solver::SciMLBase.AbstractODEAlgorithm,
        group_size::Integer; continuity_term::Real = 100, inside, kwargs...) where {F, C}
    datasize = size(ode_data, ndims(ode_data))
    griddims = ntuple(_ -> Colon(), ndims(ode_data) - 1)

    if group_size < 2 || group_size > datasize
        throw(DomainError(group_size, "group_size can't be < 2 or > number of data points"))
    end

    ranges = group_ranges(datasize, group_size)

    sols = [solve(
                remake(prob; p, tspan = (tsteps[first(rg)], tsteps[last(rg)]),
                    u0 = ode_data[griddims..., first(rg)]),
                solver;
                saveat = tsteps[rg],
                kwargs...) for rg in ranges]
    group_predictions = CuArray.(sols)

    retcodes = [sol.retcode for sol in sols]
    all(SciMLBase.successful_retcode, retcodes) || return Inf, group_predictions

    loss = 0
    for (i, rg) in enumerate(ranges)
        u = ode_data[griddims..., rg]
        û = group_predictions[i][griddims..., :]
        loss += loss_function(u, û)

        if i > 1
            loss += continuity_term *
                    continuity_loss(
                group_predictions[i - 1][griddims..., end], u[griddims..., 1])
        end
    end

    return loss, group_predictions
end

function create_loss_post_lux(
        rhs, griddims, inside, dt;
        ensemble = false,
        force_cpu = false,
        multiple_shooting = 0,
        sciml_solver = Tsit5(),
        kwargs...)
    function ArrayType()
        if force_cpu
            return Array
        end
        return CUDA.functional() ? CUDA.CuArray : Array
    end
    _loss(x, y) = Lux.MSELoss()(x[inside..., :, :], y[inside..., :, :])
    _continuity_loss(x, y) = Lux.MAELoss()(x[inside..., :], y[inside..., :])

    function _multishooting(model, ps, st, (all_u, all_t))
        uref, x, tspan, saveat_times = nothing, nothing, nothing, nothing

        CUDA.allowscalar() do
            uref = all_u[griddims..., :, 1, :]
            x = all_u[griddims..., :, 1, 1]
            saveat_times = Array(all_t[1, :])
            tspan = (all_t[1, 1], all_t[1, end])
        end

        prob = ODEProblem(rhs, x, tspan, ps)

        loss,
        _ = mymultiple_shoot(ps, uref, saveat_times, prob, _loss,
            _continuity_loss, sciml_solver, multiple_shooting;
            continuity_term = 100, adaptive = true, save_start = false, inside = inside)

        return loss, st, (; y_pred = nothing)
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

    if multiple_shooting > 0
        @info "Using multishooting loss function on overlapping intervals of size $(multiple_shooting)."
        return _multishooting
    elseif !ensemble
        @info "Using single-sample loss function"
        return _loss_function
    else
        @info "Using ensemble loss function"
        return _loss_ensemble
    end
end
