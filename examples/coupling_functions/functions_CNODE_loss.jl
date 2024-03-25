"""
    mean_squared_error(f, st, x, y, θ, λ)

Random a priori loss function. Use this function to train the closure term to reproduce the right hand side.

# Arguments:
- `f`: Function that represents the model.
- `st`: State of the model.
- `x`: Input data.
- `y`: Target data.
- `θ`: Parameters of the model.
- `λ`: Regularization parameter.

# Returns:
- `total_loss`: Mean squared error loss.
"""
function mean_squared_error(f, st, x, y, θ, λ)
    prediction = Array(f(x, θ, st)[1])
    total_loss = sum(abs2, prediction - y) / sum(abs2, y)
    return total_loss + λ * norm(θ, 1), nothing
end

"""
    create_randloss_derivative(GS_data, FG_target, f, st; nuse = size(GS_data, 2), λ=0)

Create a randomized loss function that compares the derivatives.
This function creates a randomized loss function derivative by selecting a subset of the data. This is done because using the entire dataset at each iteration would be too expensive.

# Arguments
- `GS_data`: The input data.
- `FG_target`: The target data.
- `f`: The model function.
- `st`: The model state.
- `nuse`: The number of samples to use for the loss function. Defaults to the size of `GS_data`.
- `λ`: The regularization parameter. Defaults to 0.

# Returns
A function `randloss` that computes the mean squared error loss using the selected subset of data.
"""
function create_randloss_derivative(GS_data,
        FG_target,
        f,
        st;
        nuse = size(GS_data, 2),
        λ = 0)
    d = ndims(GS_data)
    nsample = size(GS_data, d)
    function randloss(θ)
        i = Zygote.@ignore sort(shuffle(1:nsample)[1:nuse])
        x_use = Zygote.@ignore ArrayType(selectdim(GS_data, d, i))
        y_use = Zygote.@ignore ArrayType(selectdim(FG_target, d, i))
        mean_squared_error(f, st, x_use, y_use, θ, λ)
    end
end

# auxiliary function to solve the NeuralODE, given parameters p
function predict_u_CNODE(uv0, θ, tg)
    sol = Array(training_CNODE(uv0, θ, st)[1])
    #tg_size = size(tg)
    #println("sol size ", size(sol))
    #println("tg size ", size(tg))
    #println(sol[:,:,1] == tg[:,:,1])

    ## handle unstable solver
    #if any(isnan, sol)
    #    # if some steps succesfully run, then use them for the loss
    #    nok = 1
    #    while any(isnan, sol[:, :, 1:nok])
    #        nok += 1
    #    end
    #    if nok > 1
    #        println("Unstability after ", nok, " steps")
    #        tg = tg[:,:,1:nok]
    #        sol = sol[:,:,1:nok]
    #    else
    #        # otherwise run the auxiliary solver
    #        println("Using auxiliary solver ")
    #        sol = Array(training_CNODE_2(uv0, θ, st)[1])
    #        if any(isnan, sol)
    #            println("ERROR: NaN detected in the prediction")
    #            return fill(1e6 * sum(θ), tg_size)
    #        end
    #    end
    #end
    return sol, tg[:, :, 1:size(sol, 3)]
end
"""
    create_randloss_MulDtO(target; nunroll, nintervals=1, nsamples, λ_c, λ_l1)

This function creates a random loss function for the multishooting method with multiple shooting intervals.

# Arguments
- `target`: The target data for the loss function.
- `nunroll`: The number of time steps to unroll.
- `noverlaps`: The number of time steps that overlaps between each consecutive intervals.
- `nintervals`: The number of shooting intervals.
- `nsamples`: The number of samples to select.
- `λ_c`: The weight for the continuity term. It sets how strongly we make the pieces match (continuity term).
- `λ_l1`: The coefficient for the L1 regularization term in the loss function.

# Returns
- `randloss_MulDtO`: A random loss function for the multishooting method.
"""
function create_randloss_MulDtO(target; nunroll, nintervals = 1, noverlaps = 1, nsamples, λ_c, λ_l1)
    # TODO: there should be some check about the consistency of the input arguments
    # Get the number of time steps 
    d = ndims(target)
    nt = size(target, d)
    function randloss_MulDtO(θ)
        # We calculate what 
        starting_points = [i*(nunroll+1-noverlaps) for i in 1:(nintervals-1)]
        pushfirst!(starting_points,1)
        println(starting_points)
        desired_starts = []
        # We compute the requested length of consecutive timesteps
        # Notice that each interval is long nunroll+1 because we are including the initial conditions as step_0 
        length_required = (nunroll+1)*nintervals - (noverlaps+1)*(nintervals-1) 
        length_required = nunroll*nintervals - noverlaps*(nintervals-1) + nintervals
        length_required = starting_points[end] + nunroll+1
        println(length_required)
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
        θ;
        λ_c = 1e1,
        λ_l1 = 1e1,
        nunroll,
        nintervals,
        noverlaps,
        nsamples = nsamples)
    starting_points = [i*(nunroll+1-noverlaps) for i in 1:(nintervals-1)]
    pushfirst!(starting_points,1)
    println(starting_points)
    # Take all the time intervals and concatenate them in the batch dimension
    #list_tr = cat([trajectory[:, :, (1 + (i - 1) * nunroll):(1 + i * nunroll)]
    #               for i in 1:nintervals]...,
    #    dims = 2)
    for i in starting_points
        println(i:i+nunroll)
    end
    list_tr = cat([trajectory[:, :,i:(i + nunroll)]
                   for i in starting_points]...,
        dims = 2)
    # get all the initial conditions 
    #list_starts = cat([trajectory[:, :, 1 + (i - 1) * nunroll] for i in 1:nintervals]...,
    #    dims = 2)
    list_starts = cat([trajectory[:, :, i] for i in starting_points]...,
        dims = 2)
    println(size(list_tr))
    # get the predictions
    pred, list_tr = predict_u_CNODE(list_starts, θ, list_tr)
    # the loss is the sum of the differences between the real trajectory and the predicted one
    loss = sum(abs2, list_tr .- pred) ./ sum(abs2, list_tr)

    # The continuity term below is not correct because the consecutive intervals are not overlapping
    #if λ_c > 0 && size(list_tr, 3) == nunroll + 1
    #    # //TODO check if the continuity term is correct
    #    # Compute the continuity term by comparing end of one interval with the start of the next one
    #    pred_end = pred[:, :, end]
    #    pred_start = pred[:, :, 1]
    #    continuity = 0
    #    # loop over all the samples
    #    for s in 1:nsamples
    #        # each sample contains nintervals, we need to shift the index by
    #        s_shift = (s - 1) * nintervals
    #        # loop over all the intervals for the sample (excluding the last one)
    #        for i in 1:(nintervals - 1)
    #            continuity += sum(abs,
    #                pred_end[:, s_shift + i] .- pred_start[:, s_shift + i + 1])
    #        end
    #    end
    #else
    #    continuity = 0
    #end 
    if λ_c > 0 && size(list_tr, 3) == nunroll + 1
        # //TODO check if the continuity term is correct
        # Compute the continuity term by comparing end of one interval with the start of the next one
        println(size(trajectory))
        println(size(pred))
        # remind that pred[grid, (nintervals*nsamples), nunroll+1]
        pred_end = pred[:, :, end-noverlaps+1:end]
        pred_start = pred[:, :, 1:noverlaps]
        println(size(pred_end))
        println(size(pred_start))
        pred_end = pred[:, :, end]
        pred_start = pred[:, :, 1]
        println(size(pred_end))
        println(size(pred_start))
        continuity = 0
        # loop over all the samples, which have been concatenated in dim 2
        for s in 1:nsamples
            # each sample contains nintervals, we need to shift the index by
            s_shift = (s - 1) * nintervals
            #### loop over all the intervals for the sample (excluding the last one)
            for i in 1:(nintervals - 1)
                continuity += sum(abs,
                    pred_end[:, s_shift + i] .- pred_start[:, s_shift + i + 1])
            end
            ## then we loop over all the consecutive intervals to check the continuity
            #for i in starting_points
            #    continuity += sum(abs,
            #        pred_end[:, s_shift + i] .- pred_start[:, s_shift + i + 1])
            #end
        end
    else
        continuity = 0
    end

    return loss + (continuity * λ_c) + λ_l1 * norm(θ), nothing
end
