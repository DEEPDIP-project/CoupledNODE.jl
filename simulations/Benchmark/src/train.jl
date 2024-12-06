
function getdatafile(outdir, nles, filter, seed)
    joinpath(outdir, "data", splatfileparts(; seed = repr(seed), filter, nles) * ".jld2")
end

"Create data files."
createdata(; params, seeds, outdir, taskid) =
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
        data = create_les_data(; params..., rng = Xoshiro(seed), filenames, Δt = params.Δt)
        @info("Trajectory info:",
            data[1].comptime/60,
            length(data[1].t),
            Base.summarysize(data)*1e-9,)
    end

function getpriorfile(outdir, nles, filter)
    joinpath(outdir, "priortraining", splatfileparts(; filter, nles) * ".jld2")
end

"Load a-priori training results from correct file names."
loadprior(outdir, nles, filters) = map(
    splat((nles, Φ) -> load_object(getpriorfile(outdir, nles, Φ))),
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
        θ_start,
        st,
        opt,
        batchsize,
        loadcheckpoint = false,
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
        priorfile = getpriorfile(outdir, nles, Φ)
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
        io_train = CoupledNODE.NavierStokes.create_io_arrays_priori(data_train, setup)
        io_valid = CoupledNODE.NavierStokes.create_io_arrays_priori(data_valid, setup)
        θ = device(copy(θ_start))
        dataloader_prior = CoupledNODE.NavierStokes.create_dataloader_prior(
            io_train[itotal]; batchsize = batchsize,
            rng = Random.Xoshiro(dns_seeds_train[itotal]))
        train_data_priori = dataloader_prior()
        loss_priori_lux(closure, θ, st, train_data_priori)
        loss = loss_priori_lux

        callbackstate, callback = CoupledNODE.create_callback(
            closure, θ, io_valid[itotal], loss, st, batch_size = batchsize,
            rng = Xoshiro(batchseed), do_plot = true, plot_train = true)

        l, trainstate = CoupledNODE.train(
            closure, θ, st, dataloader_prior, loss; nepochs = nepoch,
            alg = opt, cpu = params.backend == CPU(), callback = callback)
        # TODO CoupledNODE has no checkpoints yet, but here it should save them
        # TODO CoupledNODE should also save some figures

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
        st,
        opt,
        nunroll_valid,
        nepoch,
        dt
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
        io_train = CoupledNODE.NavierStokes.create_io_arrays_posteriori(data_train, setup)
        io_valid = CoupledNODE.NavierStokes.create_io_arrays_posteriori(data_valid, setup)

        #θ = copy(θ_start)
        θ = device(copy(θ_start[itotal]))
        dataloader_post = CoupledNODE.NavierStokes.create_dataloader_posteriori(
            io_train[itotal]; nunroll = nunroll,
            rng = Random.Xoshiro(dns_seeds_train[itotal]))

        dudt_nn = create_right_hand_side_with_closure(
            setup[1], psolver, closure, st)
        loss = create_loss_post_lux(dudt_nn; sciml_solver = Tsit5(), dt = dt)

        callbackstate, callback = CoupledNODE.create_callback(
            closure, θ, io_valid[itotal], loss, st, nunroll = nunroll_valid,
            rng = Xoshiro(postseed), do_plot = true, plot_train = true)

        l, trainstate = CoupledNODE.train(
            closure, θ, st, dataloader_post, loss; nepochs = nepoch,
            alg = opt, cpu = params.backend == CPU(), callback = callback)
        # TODO CoupledNODE has no checkpoints yet, but here it should save them
        # TODO CoupledNODE should also save some figures

        θ = callbackstate.θmin # Use best θ instead of last θ
        results = (; θ = Array(θ), comptime = time() - starttime,
            lhist_val = callbackstate.lhist_val)
        save_object(postfile, results)
    end
    @info "Finished a-posteriori training."
end

function getsmagfile(outdir, nles, filter, projectorder)
    joinpath(outdir, "smagorinsky", splatfileparts(; projectorder, filter, nles) * ".jld2")
end

function loadsmagorinsky(outdir, nles, filters, projectorders)
    map(
        splat((nles, Φ, o) -> load_object(getsmagfile(outdir, nles, Φ, o))),
        Iterators.product(nles, filters, projectorders)
    )
end

function trainsmagorinsky(;
        params,
        projectorders,
        outdir,
        dns_seeds_train,
        taskid,
        nunroll,
        nsubstep,
        ninfo,
        θrange
)
    device(x) = adapt(params.backend, x)
    itotal = 0
    for projectorder in projectorders, Φ in params.filters, nles in params.nles
        itotal += 1
        if isnothing(taskid) || itotal == taskid
            @info "Training Smagorinsky" projectorder Φ nles
        else
            # Each task does one training
            @info "Skipping Smagorinsky training for iteration $(itotal) != $(taskid)"
            continue
        end
        starttime = time()
        T = typeof(params.Re)
        smagfile = getsmagfile(outdir, nles, Φ, projectorder)
        smagdir = dirname(smagfile)
        ispath(smagdir) || mkpath(smagdir)
        setup = getsetup(; params, nles)
        psolver = default_psolver(setup)
        d = namedtupleload(getdatafile(outdir, nles, Φ, dns_seeds_train[1]))
        it = 1:nunroll
        data = (; u = selectdim(d.u, ndims(d.u), it) |> collect |> device, t = d.t[it])
        θmin = T(0)
        emin = T(Inf)
        err = create_relerr_post(;
            data,
            setup,
            psolver,
            method = RKProject(params.method, projectorder),
            closure_model = IncompressibleNavierStokes.smagorinsky_closure_natural(setup),
            nupdate = nsubstep # Number of time steps between t[i] and t[i + 1]
        )
        for (iθ, θ) in enumerate(θrange)
            iθ % ninfo == 0 && @info "Testing θ = $θ"
            e = err(θ)
            if e < emin
                emin = e
                θmin = θ
            end
        end
        @info "Optimal θ = $θmin"
        results = (; θ = θmin, comptime = time() - starttime)
        save_object(smagfile, results)
    end
    @info "Finished Smagorinsky training."
end
