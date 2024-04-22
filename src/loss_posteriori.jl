import Zygote
import Random: shuffle
import LinearAlgebra: norm
# *********************
# DtO Loss for NeuralODE object
function create_randloss_DtO(ubar; nunroll)
    d = ndims(ubar)
    nt = size(ubar, d)
    function randloss_DtO(p)
        # Zygote will select a random initial condition of lenght nunroll
        istart = Zygote.@ignore rand(1:(nt - nunroll))
        trajectory = Zygote.@ignore ArrayType(selectdim(ubar, d, istart:(istart + nunroll)))
        # this is the loss evaluated for each piece
        loss_DtO_onepiece(trajectory, p)
    end
end
# Piecewise loss function
function loss_DtO_onepiece(trajectory, p)
    #tr_start = trajectory[:,:,:,1]
    tr_start = [trajectory[1]]
    pred = predict_NODE(tr_start, p)
    loss = sum(abs2, trajectory .- pred) ./ sum(abs2, trajectory)
    return loss, pred
end

# auxiliary function to solve the NeuralODE, given parameters p
function predict_NODE(u0, θ, st, training_NODE)
    # you can make this shorter, since it is used only for the unrolling
    return Array(training_NODE(u0, θ, st)[1])
end

# *********************
# Multishooting DtO loss for NeuralODE object - should be removed
function create_randloss_MulDtO_1(ubar, training_NODE, st; nunroll, nintervals)
    d = ndims(ubar)
    nt = size(ubar, d)
    function randloss_MulDtO(θ)
        # Zygote will select a random initial condition that can accomodate all the multishooting intervals
        istart = Zygote.@ignore rand(1:(nt - nunroll * nintervals))
        trajectory = Zygote.@ignore ArrayType(selectdim(ubar,
            d,
            istart:(istart + nunroll * nintervals)))
        # this is the loss evaluated for each multishooting set
        loss_MulDtO_oneset_1(
            trajectory, θ, st, training_NODE, nunroll = nunroll, nintervals = nintervals)
    end
end
# the parameter λ sets how strongly we make the pieces match (continuity term)
function loss_MulDtO_oneset_1(
        trajectory, θ, st, training_NODE; λ = 1e3, nunroll, nintervals)
    loss = 0.0
    last_pred = nothing
    for i in 1:nintervals
        tr_start = [trajectory[1 + (i - 1) * nunroll]]
        # Warning: hard coded dimension! 
        #tr_start = trajectory[:,:,:,1+(i-1)*nunroll]
        # hard coded single batch
        pred = dropdims(predict_NODE(tr_start, θ, st, training_NODE), dims = 1)
        real_tr = trajectory[(1 + (i - 1) * nunroll):(1 + i * nunroll)]
        # if they have different lenghts, make them match to the shortest
        if length(pred) > length(real_tr)
            pred = pred[1:length(real_tr)]
        end
        if length(pred) < length(real_tr)
            real_tr = real_tr[1:length(pred)]
        end
        loss += sum(abs2, real_tr .- pred) ./ sum(abs2, real_tr)
        #loss += sum(abs2, trajectory[1+(i-1)*nunroll:1+i*nunroll] .- pred) ./ sum(abs2, trajectory[1+(i-1)*nunroll:1+i*nunroll])
        #loss += sum(abs2, trajectory[:,:,:,1+(i-1)*nunroll:1+i*nunroll] .- pred) ./ sum(abs2, trajectory[:,:,:,1+(i-1)*nunroll:1+i*nunroll])
        # add continuity term
        if last_pred !== nothing
            loss += λ * sum(abs, last_pred .- tr_start)
        end
        last_pred = pred[end]
    end
    return loss, nothing
end

# auxiliary function to solve the NeuralODE, given parameters p
function predict_u_CNODE(uv0, θ, st, nunroll, training_CNODE, tg)
    sol = Array(training_CNODE(uv0, θ, st)[1])
    return sol[:, :, 1:nunroll], tg[:, :, 1:nunroll]
end

"""
    create_randloss_MulDtO(target; nunroll, nintervals=1, nsamples, λ_c, λ_l1)

Creates a random loss function for the multishooting method with multiple shooting intervals.

# Arguments
- `target`: The target data for the loss function.
- `training_CNODE`: Model CNODE.
- `nunroll`: The number of time steps to unroll.
- `noverlaps`: The number of time steps that overlaps between each consecutive intervals.
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
    nt = size(target, d)
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
- `nsamples`: The number of samples. Default is `nsamples`.

# Returns
- `loss`: The computed loss value.
- `nothing`: Placeholder return value.
"""
function loss_MulDtO_oneset(trajectory,
        θ, st,
        training_CNODE;
        λ_c = 1e1,
        λ_l1 = 1e1,
        nunroll,
        nintervals,
        noverlaps,
        nsamples = nsamples)
    # Get the timesteps where the intervals start 
    starting_points = [i == 0 ? 1 : i * (nunroll + 1 - noverlaps)
                       for i in 0:(nintervals - 1)]
    # Take all the time intervals and concatenate them in the batch dimension
    list_tr = cat([trajectory[:, :, i:(i + nunroll)]
                   for i in starting_points]...,
        dims = 2)
    # Get all the initial conditions 
    list_starts = cat([trajectory[:, :, i] for i in starting_points]...,
        dims = 2)
    # Use the differentiable solver to get the predictions
    pred, target = predict_u_CNODE(list_starts, θ, st, nunroll, training_CNODE, list_tr)
    # the loss is the sum of the differences between the real trajectory and the predicted one
    loss = sum(abs2, target[:, :, 2:nunroll] .- pred[:, :, 2:nunroll]) ./
           sum(abs2, target[:, :, 2:nunroll])

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
        # loop over all the samples, which have been concatenated in dim 2
        for s in 1:nsamples
            # each sample contains nintervals, we need to shift the index by
            s_shift = (s - 1) * nintervals
            # loop over all the intervals for the sample (excluding the last one)
            for i in 1:(nintervals - 1)
                continuity += sum(abs,
                    pred_end[:, s_shift + i] .- pred_start[:, s_shift + i + 1])
            end
        end
    else
        continuity = 0
    end

    return loss + (continuity * λ_c) + λ_l1 * norm(θ), nothing
end
