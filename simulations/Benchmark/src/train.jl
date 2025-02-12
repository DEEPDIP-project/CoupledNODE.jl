
function getdatafile(outdir, nles, filter, seed)
    joinpath(outdir, "data", splatfileparts(; seed = repr(seed), filter, nles) * ".jld2")
end

"Create data files."
createdata(; params, seeds, outdir, taskid, backend) =
    for (iseed, seed) in enumerate(seeds)
        if isnothing(taskid) || iseed == taskid
            @info "Creating DNS trajectory for seed $(repr(seed))"
        else
            # Each task does one initial condition
            @info "Skipping seed $(repr(seed)) for task $taskid"
            continue
        end
        filenames = []
        for (nles, Φ) in Iterators.product(params.nles, params.filters)
            f = getdatafile(outdir, nles, Φ, seed)
            datadir = dirname(f)
            ispath(datadir) || mkpath(datadir)
            push!(filenames, f)
        end
        if isfile(filenames[1])
            @info "Data file $(filenames[1]) already exists. Skipping."
            continue
        end
        data = create_les_data(;
            params..., rng = Xoshiro(seed), filenames, Δt = params.Δt, backend = backend)
        @info("Trajectory info:",
            data[1].comptime/60,
            length(data[1].t),
            Base.summarysize(data)*1e-9,)
    end

function getpriorfile(outdir, closure_name, nles, filter)
    joinpath(
        outdir, "priortraining", closure_name, splatfileparts(; filter, nles) * ".jld2")
end

"Load a-priori training results from correct file names."
loadprior(outdir, closure_name, nles, filters) = map(
    splat((nles, Φ) -> load_object(getpriorfile(outdir, closure_name, nles, Φ))),
    Iterators.product(nles, filters)
)

"Train with a-priori loss."
function trainprior(;
        params,
        priorseed,
        dns_seeds_train,
        dns_seeds_valid,
        taskid,
        outdir,
        plotdir,
        closure,
        closure_name,
        θ_start,
        st,
        opt,
        batchsize,
        loadcheckpoint = true,
        do_plot = false,
        plot_train = false,
        nepoch
)
    device(x) = adapt(params.backend, x)
    itotal = 0
    for Φ in params.filters, nles in params.nles
        itotal += 1
        if isnothing(taskid) || itotal == taskid
            @info "Training a-priori" Φ nles
        else
            # Each task does one training
            @info "Skipping a-priori training for iteration $(itotal) != $(taskid)"
            continue
        end
        starttime = time()
        priorfile = getpriorfile(outdir, closure_name, nles, Φ)
        priordir = dirname(priorfile)
        ispath(priordir) || mkpath(priordir)
        figdir = joinpath(plotdir, "priortraining")
        ispath(figdir) || mkpath(figdir)
        figfile = joinpath(figdir, splitext(basename(priorfile))[1] * ".pdf")
        checkfile = join(splitext(priorfile), "_checkpoint")
        batchseed, validseed = splitseed(priorseed, 2) # Same seed for all training setups

        # Read the data in the format expected by the CoupledNODE
        T = eltype(params.Re)
        setup = []
        for nl in nles
            x = ntuple(α -> LinRange(T(0.0), T(1.0), nl + 1), params.D)
            push!(setup, Setup(; x = x, Re = params.Re))
        end

        # Read the data in the format expected by the CoupledNODE
        data_train = []
        for s in dns_seeds_train
            data_i = namedtupleload(getdatafile(outdir, nles, Φ, s))
            push!(data_train, hcat(data_i))
        end
        data_valid = []
        for s in dns_seeds_valid
            data_i = namedtupleload(getdatafile(outdir, nles, Φ, s))
            push!(data_valid, hcat(data_i))
        end
        NS = Base.get_extension(CoupledNODE, :NavierStokes)
        io_train = NS.create_io_arrays_priori(data_train, setup)
        io_valid = NS.create_io_arrays_priori(data_valid, setup)
        θ = device(copy(θ_start))
        dataloader_prior = NS.create_dataloader_prior(
            io_train[itotal]; batchsize = batchsize,
            rng = Random.Xoshiro(dns_seeds_train[itotal]), device = device)
        train_data_priori = dataloader_prior()
        @info device
        @info typeof(train_data_priori)
        loss_priori_lux(closure, θ, st, train_data_priori)
        loss = loss_priori_lux

        if loadcheckpoint && isfile(checkfile)
            callbackstate, trainstate, epochs_trained = CoupledNODE.load_checkpoint(checkfile)
            nepochs_left = nepoch - epochs_trained
        else
            callbackstate = trainstate = nothing
            nepochs_left = nepoch
        end

        callbackstate, callback = NS.create_callback(
            closure, θ, io_valid[itotal], loss, st;
            callbackstate = callbackstate, batch_size = batchsize,
            rng = Xoshiro(batchseed), do_plot = do_plot, plot_train = plot_train, figfile = figfile, device = device)

        if nepochs_left <= 0
            @info "No epochs left to train."
            continue
        else
            l, trainstate = CoupledNODE.train(
                closure, θ, st, dataloader_prior, loss; tstate = trainstate,
                nepochs = nepochs_left,
                alg = opt, cpu = !CUDA.functional(), callback = callback)
        end
        save_object(checkfile, (callbackstate = callbackstate, trainstate = trainstate))

        θ = callbackstate.θmin # Use best θ instead of last θ
        results = (; θ = Array(θ), comptime = time() - starttime,
            callbackstate.lhist_val, callbackstate.lhist_nomodel)
        save_object(priorfile, results)
    end
    @info "Finished a-priori training."
end

function getpostfile(outdir, nles, filter, projectorder)
    joinpath(outdir, "posttraining", splatfileparts(; projectorder, filter, nles) * ".jld2")
end

"Load a-posteriori training results from correct file names."
loadpost(outdir, nles, filters, projectorders) = map(
    splat((nles, Φ, o) -> load_object(getpostfile(outdir, nles, Φ, o))),
    Iterators.product(nles, filters, projectorders)
)

"Train with a-posteriori loss function."
function trainpost(;
        params,
        projectorders,
        outdir,
        plotdir,
        taskid,
        postseed,
        dns_seeds_train,
        dns_seeds_valid,
        nunroll,
        closure,
        θ_start,
        loadcheckpoint = true,
        st,
        opt,
        nunroll_valid,
        nepoch,
        dt,
        do_plot = false,
        plot_train = false
)
    device(x) = adapt(params.backend, x)
    itotal = 0
    for projectorder in projectorders,
        (ifil, Φ) in enumerate(params.filters),
        (igrid, nles) in enumerate(params.nles)

        itotal += 1
        if isnothing(taskid) || itotal == taskid
            @info "Training a-posteriori" projectorder Φ nles
        else
            # Each task does one training
            @info "Skipping a-posteriori training for iteration $(itotal) != $(taskid)"
            continue
        end
        starttime = time()
        postfile = getpostfile(outdir, nles, Φ, projectorder)
        ispath(dirname(postfile)) || mkpath(dirname(postfile))
        figdir = joinpath(plotdir, "posttraining")
        ispath(figdir) || mkpath(figdir)
        figfile = joinpath(figdir, splitext(basename(postfile))[1] * ".pdf")
        checkfile = join(splitext(postfile), "_checkpoint")
        setup = getsetup(; params, nles)
        psolver = default_psolver(setup)
        # Read the data in the format expected by the CoupledNODE
        T = eltype(params.Re)
        setup = []
        for nl in nles
            x = ntuple(α -> LinRange(T(0.0), T(1.0), nl + 1), params.D)
            push!(setup, Setup(; x = x, Re = params.Re, params.backend))
        end

        # Read the data in the format expected by the CoupledNODE
        data_train = []
        for s in dns_seeds_train
            data_i = namedtupleload(getdatafile(outdir, nles, Φ, s))
            push!(data_train, hcat(data_i))
        end
        data_valid = []
        for s in dns_seeds_valid
            data_i = namedtupleload(getdatafile(outdir, nles, Φ, s))
            push!(data_valid, hcat(data_i))
        end

        NS = Base.get_extension(CoupledNODE, :NavierStokes)
        io_train = NS.create_io_arrays_posteriori(data_train, setup)
        io_valid = NS.create_io_arrays_posteriori(data_valid, setup)
        θ = device(copy(θ_start[itotal]))
        dataloader_post = NS.create_dataloader_posteriori(
            io_train[itotal]; nunroll = nunroll,
            rng = Random.Xoshiro(dns_seeds_train[itotal]), device = device)

        dudt_nn = NS.create_right_hand_side_with_closure(
            setup[1], psolver, closure, st)
        loss = create_loss_post_lux(
            dudt_nn; sciml_solver = Tsit5(), dt = dt, use_cuda = CUDA.functional())

        if loadcheckpoint && isfile(checkfile)
            callbackstate, trainstate, epochs_trained = CoupledNODE.load_checkpoint(checkfile)
            nepochs_left = nepoch - epochs_trained
        else
            callbackstate = trainstate = nothing
            nepochs_left = nepoch
        end

        callbackstate, callback = NS.create_callback(
            closure, θ, io_valid[itotal], loss, st;
            callbackstate = callbackstate, nunroll = nunroll_valid,
            rng = Xoshiro(postseed), do_plot = do_plot, plot_train = plot_train, figfile = figfile, device = device)
        if nepochs_left <= 0
            @info "No epochs left to train."
            continue
        else
            l, trainstate = CoupledNODE.train(
                closure, θ, st, dataloader_post, loss; tstate = trainstate, nepochs = nepochs_left,
                alg = opt, cpu = !CUDA.functional(), callback = callback)
        end
        save_object(checkfile, (callbackstate = callbackstate, trainstate = trainstate))

        θ = callbackstate.θmin # Use best θ instead of last θ
        results = (; θ = Array(θ), comptime = time() - starttime,
            lhist_val = callbackstate.lhist_val)
        save_object(postfile, results)
    end
    @info "Finished a-posteriori training."
end
