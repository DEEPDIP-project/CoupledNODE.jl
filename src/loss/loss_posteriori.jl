import Zygote
import Random: shuffle
import LinearAlgebra: norm
import DifferentialEquations: ODEProblem, solve, Tsit5

"""
    predict_u_CNODE(uv0, θ, st, nunroll, training_CNODE, tg)

Auxiliary function to solve the NeuralODE. Returns the prediction and the target sliced to nunroll time steps for convenience when calculating the loss.

# Arguments
- `uv0`: Initial condition(s) for the NeuralODE.
- `θ`: Parameters of the NeuralODE.
- `st`: Time steps for the NeuralODE.
- `nunroll`: Number of time steps in the window.
- `training_CNODE`: CNODE model that solves the NeuralODE.
- `tg`: Target values.

# Returns
- `sol`: Prediction of the NeuralODE sliced to nunroll time steps.
- `tg[:, :, 1:nunroll]`: Target values sliced to nunroll time steps.
"""
function predict_u_CNODE(uv0, θ, st, nunroll, training_CNODE, tg)
    sol = Array(training_CNODE(uv0, θ, st)[1])
    return sol, tg
end

"""
    create_randloss_MulDtO(target; nunroll, nintervals=1, nsamples, λ_c, λ_l1)

Creates a random loss function for the multishooting method with multiple shooting intervals.

# Arguments
- `target`: The target data for the loss function.
- `training_CNODE`: Model CNODE.
- `st`: state of the neural part.
- `nunroll`: The number of time steps to unroll.
- `noverlaps`: The number of time steps that overlaps between consecutive intervals.
- `nintervals`: The number of shooting intervals.
- `nsamples`: The number of samples to select.
- `λ_c`: The weight for the continuity term. It sets how strongly we make the pieces match (continuity term).
- `λ_l1`: The coefficient for the L1 regularization term in the loss function.

# Returns
- `randloss_MulDtO`: A random loss function for the multishooting method.
"""
function create_randloss_MulDtO(
        target, training_CNODE, st; nunroll, nintervals = 1,
        noverlaps = 1, nsamples, λ_c = 1e2, λ_l1 = 1e-1)
    # TODO: there should be some check about the consistency of the input arguments
    # Get the number of time steps 
    d = ndims(target)
    nt = size(target, d) # number of time steps (length of last dimension)
    function randloss_MulDtO(θ)
        # Compute the requested length of consecutive timesteps
        # Notice that each interval is long nunroll+1 because we are including the initial conditions as step_0 
        length_required = nintervals * (nunroll + 1) - noverlaps * (nintervals - 1)
        # Zygote will select a random initial condition that can accomodate all the multishooting intervals
        istart = Zygote.@ignore rand(1:(nt - length_required))
        trajectory = Zygote.@ignore ArrayType(selectdim(target,
            d,
            istart:(istart + length_required)))
        # and select a certain number of samples
        trajectory = Zygote.@ignore trajectory[:, rand(1:size(trajectory, 2), nsamples), :]
        # then return the loss for each multishooting set
        loss_MulDtO_oneset(trajectory,
            θ,
            st,
            training_CNODE,
            nunroll = nunroll,
            nintervals = nintervals,
            noverlaps = noverlaps,
            nsamples = nsamples,
            λ_c = λ_c,
            λ_l1 = λ_l1)
    end
end

"""
    loss_MulDtO_oneset(trajectory, θ; λ_c=1e1, λ_l1=1e1, nunroll, nintervals, nsamples=nsamples)

Compute the loss function for the multiple shooting method with a continuous neural ODE (CNODE) model.
Check https://docs.sciml.ai/DiffEqFlux/dev/examples/multiple_shooting/ for more details.

# Arguments
- `trajectory`: The trajectory of the system.
- `θ`: The parameters of the CNODE model.
- `training_CNODE`: Model CNODE.
- `λ_c`: The weight for the continuity term. It sets how strongly we make the pieces match (continuity term). Default is `1e1`.
- `λ_l1`: The weight for the L1 regularization term. Default is `1e1`.
- `nunroll`: The number of time steps to unroll the trajectory.
- `noverlaps`: The number of time steps that overlaps between each consecutive intervals.
- `nintervals`: The number of intervals to divide the trajectory into.
- `nsamples`: The number of samples.

# Returns
- `loss`: The computed loss value.
"""
function loss_MulDtO_oneset(trajectory,
        θ, st,
        training_CNODE;
        λ_c = 1e1,
        λ_l1 = 1e1,
        nunroll,
        nintervals,
        noverlaps,
        nsamples)
    # Get the timesteps where the intervals start 
    starting_points = [i == 0 ? 1 : i * (nunroll + 1 - noverlaps)
                       for i in 0:(nintervals - 1)]
    # Take all the time intervals and concatenate them in the batch dimension
    # [!] notice that this shuffles the datasamples, so along the new dimension 2 we will group by starting point 
    list_tr = cat([trajectory[:, :, i:(i + nunroll)]
                   for i in starting_points]...,
        dims = 2)
    # Get all the initial conditions 
    list_starts = list_tr[:, :, 1]
    # Use the differentiable solver to get the predictions
    pred, target = predict_u_CNODE(list_starts, θ, st, nunroll, training_CNODE, list_tr)
    # the loss is the sum of the differences between the real trajectory and the predicted one
    #loss = sum(abs2, (target[:,:,2:end] .- pred[:,:,2:end] ) ./ (target[:,:,2:end].+1e-5))
    loss = sum(abs2, target[:, :, 2:end] .- pred[:, :, 2:end])

    if λ_c > 0 && size(list_tr, 3) == nunroll + 1
        # //TODO check if the continuity term is correct
        # Compute the continuity term by comparing end of one interval with the start of the next one
        # (!) Remind that the trajectory is stored as: 
        #   pred[grid, (nintervals*nsamples), nunroll+1]
        # and we need to compare the last noverlaps points of an interval
        pred_end = pred[:, :, (end - noverlaps + 1):end]
        # with the first noverlaps points of the next interval EXCLUDING the initial condition 
        # (which is already part of the loss function)
        pred_start = pred[:, :, 2:(1 + noverlaps)]
        continuity = 0
        # loop over all the datasamples, which have been concatenated in dim 2
        for s in 1:nsamples
            for x in 1:(nintervals - 1)
                # each sample contains nintervals, so we need to shift the index like this
                ini_id = s + (x - 1) * nsamples
                end_id = s + (x - 1) * nsamples + nsamples
                continuity += sum(abs, pred_end[:, ini_id, :] .- pred_start[:, end_id, :])
            end
        end
    else
        continuity = 0
    end

    return loss + (continuity * λ_c) + λ_l1 * norm(θ), nothing
end

function create_loss_post_lux(rhs; sciml_solver = Tsit5(), kwargs...)
    function loss_function(model, ps, st, (u, t))
        x = u[:, :, :, 1:1]
        y = u[:, :, :, 2:end] # remember to discard sol at the initial time step
        #dt = params.Δt
        dt = t[2] - t[1]
        #saveat_loss = [i * dt for i in 1:length(y)]
        tspan = [t[1], t[end]]
        prob = ODEProblem(rhs, x, tspan, ps)
        pred = Array(solve(
            prob, sciml_solver; u0 = x, p = ps, dt = dt, adaptive = false, kwargs...))
        # remember that the first element of pred is the initial condition (SciML)
        return sum(abs2, y - pred[:, :, :, 1, 2:end]) / sum(abs2, y), st, (; y_pred = pred)
    end
end
