using Zygote: Zygote
using CUDA: CUDA
using Random: shuffle
using LinearAlgebra: norm
using DifferentialEquations: ODEProblem, solve, Tsit5, RK4
using Lux: Lux
using ChainRulesCore: ignore_derivatives

"""
    create_loss_post_lux(rhs; sciml_solver = Tsit5(), kwargs...)

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
        rhs, griddims, inside; sciml_solver = Tsit5(), kwargs...)
    function loss_function(model, ps, st, (all_u, all_t))
        nsamp = size(all_u, ndims(all_u) - 1)
        nts = size(all_u, ndims(all_u))
        loss = 0
        for si in 1:nsamp
            uref, x, t, tspan, dt,
            prob, pred = nothing, nothing, nothing, nothing, nothing, nothing, nothing # initialize variable outside allowscalar do.

            CUDA.allowscalar() do
                uref = all_u[inside..., :, si, 2:nts]
                x = all_u[griddims..., :, si, 1]
                t = all_t[si, 2:nts]
                tspan = (all_t[si, 1], all_t[si, end])
            end
            prob = ODEProblem(rhs, x, tspan, ps)
            pred = solve(
                prob, sciml_solver; u0 = x, p = ps,
                adaptive = true, save_start = false, saveat = Array(t), kwargs...)
            if size(pred) != size(uref)
                @warn "Instability in the loss function. The predicted and target data have different sizes."
                @info "Predicted size: $(size(pred))"
                @info "Target size: $(size(uref))"
                @info "size(t): $(size(t))"
                @info "t: $(t)"
                return Inf, st, (; y_pred = pred)
            else
                loss += sum(
                    sum((pred[inside..., :, :] .- uref) .^ 2, dims = (1, 2, 3)) ./
                    sum(abs2, uref, dims = (1, 2, 3))
                ) / (nts-1)
            end
        end
        return loss/nsamp, st, (; y_pred = nothing)
    end
end

"""
    validate_results(model, p, dataloader, nuse = 100)

Validate the results of the model using the given parameters `p`, dataloader, and number of samples `nuse`.

# Arguments
- `model`: The model to evaluate.
- `p`: Parameters for the model.
- `dataloader::Function`: A function that returns the data `(u, t)`.
- `nuse::Int`: The number of samples to use for validation. Defaults to `100`.

# Returns
 - `loss`: The computed loss value.
"""
function validate_results(model, p, dataloader, nuse = 100)
    loss = 0
    for _ in 1:nuse
        data = dataloader()
        u, t = data
        griddims = axes(u)[1:(ndims(u) - 2)]
        x = u[griddims..., :, 1]
        y = u[griddims..., :, 2:end] # remember to discard sol at the initial time step
        #dt = params.Δt
        dt = t[2] - t[1]
        #saveat_loss = [i * dt for i in 1:length(y)]
        tspan = [t[1], t[end]]
        prob = ODEProblem(model, x, tspan, p)
        pred = Array(solve(prob, Tsit5(); u0 = x, p = p, dt = dt, adaptive = false))
        # remember that the first element of pred is the initial condition (SciML)
        loss += sum(
            abs2, y[griddims..., :, 1:(size(pred, 4) - 1)] - pred[griddims..., :, 2:end]) /
                sum(abs2, y)
    end
    return loss / nuse
end
