# Random a priori loss function
# You can use this to train the closure term to reproduce the right hand side
function mean_squared_error(f, st, x, y, θ, λ) 
    prediction = Array(f(x, θ, st)[1])
    #total_loss = sum(abs2, prediction - y)
    #return total_loss, nothing
    total_loss = sum(abs2, prediction - y) / sum(abs2, y)
#    println(θ)
    return total_loss + λ *norm(θ, 1), nothing
    ## Add a regularization for the parameters that are weakly used
    #reg_term = 0.0
    #for θ_i in θ
    #    if abs(θ_i) < 0.1
    #        reg_term += abs(θ_i)
    #    end
    #end
    #return total_loss + λ *reg_term, nothing
end
# the loss functions are randomized by selecting a subset of the data, because it would be too expensive to use the entire dataset at each iteration
function create_randloss_derivative(GS_data, FG_target, f, st; nuse = size(GS_data, 2), λ=0)
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
function predict_u_CNODE(uv0,θ,tg_size)
    sol = Array(training_CNODE(uv0, θ, st)[1])
    
    # handle unstable solver
    # //TODO you probably don't want to use this
    if any(isnan, sol)
        # Create a mask of the NaN values
        nan_mask = isnan.(sol)
        # Get the non-NaN values
        non_nan_values = sol[.!nan_mask]
        
        target = non_nan_values[rand(1:length(non_nan_values), prod(tg_size))] |> f32
        target = reshape(target, tg_size)
        sol = target
        
    end
    
    return sol
end
# This is the function to create the loss
function create_randloss_MulDtO(target; nunroll, nintervals=1, nsamples, λ)
    # Get the number of time steps 
    d = ndims(target)
    nt = size(target, d)
    function randloss_MulDtO(θ)
        # Zygote will select a random initial condition that can accomodate all the multishooting intervals
        istart = Zygote.@ignore rand(1:nt-nunroll*nintervals)
        trajectory = Zygote.@ignore ArrayType(selectdim(target, d, istart:istart+nunroll*nintervals))
        # and select a certain number of samples
        trajectory = Zygote.@ignore trajectory[:,rand(1:size(trajectory,2), nsamples),:]
        # this is the loss evaluated for each multishooting set
        loss_MulDtO_oneset(trajectory, θ, nunroll=nunroll, nintervals=nintervals, nsamples=nsamples, λ=λ)
    end
end
# the parameter λ sets how strongly we make the pieces match (continuity term)
function loss_MulDtO_oneset(trajectory, θ; λ=1e1, nunroll, nintervals, nsamples=nsamples)

    # Take all the time intervals and concatenate them in the batch dimension
    list_tr = cat([trajectory[:, :, 1+(i-1)*nunroll:1+i*nunroll] for i in 1:nintervals]..., dims=2)
    # get all the initial conditions 
    list_starts = cat([trajectory[:, :, 1+(i-1)*nunroll] for i in 1:nintervals]..., dims=2)
    pred = predict_u_CNODE(list_starts,θ,size(list_tr))
    # check if pred contains any nan
    if any(isnan, pred)
        println("ERROR: NaN detected in the prediction")
        #pred = list_tr .+ 1e10
        #return Inf, nothing
        println("size tr ",size(list_tr))
        println("size pred ", size(pred))
        print("\n**********\n")
    end
    # the loss is the sum of the differences between the real trajectory and the predicted one
    loss = sum(abs2, list_tr.- pred) ./ sum(abs2, list_tr)

    if λ>0
        # //TODO check if the continuity term is correct
        # Then I compute the continuity term by comparing end of one interval with the start of the next one
        pred_end = pred[:,:,end]
        pred_start = pred[:,:,1]
        continuity = 0
        # loop over all the samples
        for s in 1:nsamples
            # each sample contains nintervals, so I need to shift the index by
            s_shift = (s-1)*nintervals
            # the I loop over all the intervals for the sample (excluding the last one)
            for i in 1:nintervals-1
                continuity += sum(abs,pred_end[:, s_shift+ i] .- pred_start[:, s_shift+ i+1])
            end
        end
    else
        continuity = 0
    end

    return loss+(continuity*λ), nothing
end