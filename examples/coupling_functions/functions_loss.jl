
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
function predict_NODE(u0, θ)
    # you can make this shorter, since it is used only for the unrolling
    return Array(training_NODE(u0, θ, st)[1])
end

# *********************
# Multishooting DtO loss for NeuralODE object
function create_randloss_MulDtO(ubar; nunroll, nintervals)
    d = ndims(ubar)
    nt = size(ubar, d)
    function randloss_MulDtO(θ)
        # Zygote will select a random initial condition that can accomodate all the multishooting intervals
        istart = Zygote.@ignore rand(1:(nt - nunroll * nintervals))
        trajectory = Zygote.@ignore ArrayType(selectdim(ubar,
            d,
            istart:(istart + nunroll * nintervals)))
        # this is the loss evaluated for each multishooting set
        loss_MulDtO_oneset(trajectory, θ, nunroll = nunroll, nintervals = nintervals)
    end
end
# the parameter λ sets how strongly we make the pieces match (continuity term)
function loss_MulDtO_oneset(trajectory, θ; λ = 1e3, nunroll, nintervals)
    loss = 0.0
    last_pred = nothing
    for i in 1:nintervals
        tr_start = [trajectory[1 + (i - 1) * nunroll]]
        # Warning: hard coded dimension! 
        #tr_start = trajectory[:,:,:,1+(i-1)*nunroll]
        # hard coded single batch
        pred = dropdims(predict_NODE(tr_start, θ), dims = 1)
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
