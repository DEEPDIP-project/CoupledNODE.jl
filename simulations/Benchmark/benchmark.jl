#! format: off 
if false                      #src
    include("src/Benchmark.jl") #src
end                           #src

@info "Script started"
@info VERSION

using Pkg
@info Pkg.status()

# Color palette for consistent theme throughout paper
palette = (; color = ["#3366cc", "#cc0000", "#669900", "#ff9900"])

########################################################################## #src
# Read the configuration file
using IncompressibleNavierStokes
using NeuralClosure
using CoupledNODE
NS = Base.get_extension(CoupledNODE, :NavierStokes)
global conf
try
    conf = NS.read_config(ENV["CONF_FILE"])
    @info "Reading configuration file from ENV"
catch
    @info "Reading configuration file from default"
    conf = NS.read_config("configs/conf.yaml")
end
########################################################################## #src

# Choose where to put output
basedir = haskey(ENV, "DEEPDIP") ? ENV["DEEPDIP"] : @__DIR__
outdir = joinpath(basedir, "output", "kolmogorov")
closure_name = conf["closure"]["name"]
outdir_model = joinpath(outdir, closure_name)
plotdir = joinpath(outdir, closure_name, "plots")
logdir = joinpath(outdir, closure_name, "logs")
ispath(outdir) || mkpath(outdir)
ispath(plotdir) || mkpath(plotdir)
ispath(logdir) || mkpath(logdir)

# Turn off plots for array jobs.
# If all the workers do this at the same time, one might
# error when saving the file at the same time
doplot() = true

########################################################################## #src

# ## Configure logger

using Benchmark
using Dates

# Write output to file, as the default SLURM file is not updated often enough
isslurm = haskey(ENV, "SLURM_JOB_ID")
if isslurm
    jobid = parse(Int, ENV["SLURM_JOB_ID"])
    taskid = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
    logfile = "job=$(jobid)_task=$(taskid)_$(Dates.now()).out"
else
    taskid = nothing
    logfile = "log_$(Dates.now()).out"
end
logfile = joinpath(logdir, logfile)
setsnelliuslogger(logfile)

@info "# A-posteriori analysis: Forced turbulence (2D)"

# ## Load packages

@info "Loading packages"

using Accessors
using Adapt
using CairoMakie
using CoupledNODE: loss_priori_lux, create_loss_post_lux
using CUDA
using DifferentialEquations
using IncompressibleNavierStokes.RKMethods
using JLD2
using LaTeXStrings
using LinearAlgebra
using Lux
using LuxCUDA
using NNlib
using Optimisers
using ParameterSchedulers
using Random
using SparseArrays


# ## Random number seeds
#
# Use a new RNG with deterministic seed for each code "section"
# so that e.g. training batch selection does not depend on whether we
# generated fresh filtered DNS data or loaded existing one (the
# generation of which would change the state of a global RNG).
#
# Note: Using `rng = Random.default_rng()` twice seems to point to the
# same RNG, and mutating one also mutates the other.
# `rng = Xoshiro()` creates an independent copy each time.
#
# We define all the seeds here.

seeds = NS.load_seeds(conf)

########################################################################## #src

# ## Hardware selection

# Precision
T = eval(Meta.parse(conf["T"]))

# Device
if CUDA.functional()
    ## For running on a CUDA compatible GPU
    @info "Running on CUDA"
    backend = CUDABackend()
    CUDA.allowscalar(false)
    device = x -> adapt(CuArray, x)
    clean() = (GC.gc(); CUDA.reclaim())
else
    ## For running on CPU.
    ## Consider reducing the sizes of DNS, LES, and CNN layers if
    ## you want to test run on a laptop.
    @warn "Running on CPU"
    backend = CPU()
    device = identity
    clean() = nothing
end
conf["params"]["backend"] = deepcopy(backend)
@info backend
@info CUDA.versioninfo()


########################################################################## #src

# ## Data generation
#
# Create filtered DNS data for training, validation, and testing.

# Parameters
params = NS.load_params(conf)
@info params

# DNS seeds
ntrajectory = conf["ntrajectory"]
dns_seeds = splitseed(seeds.dns, ntrajectory)
dns_seeds_train = dns_seeds[1:ntrajectory-2]
dns_seeds_valid = dns_seeds[ntrajectory-1:ntrajectory-1]
dns_seeds_test = dns_seeds[ntrajectory:ntrajectory]

# Create data
docreatedata = conf["docreatedata"]
docreatedata && createdata(; params, seeds = dns_seeds, outdir, taskid)
@info "Data generated"

# Computational time
docomp = conf["docomp"]
docomp && let
    comptime, datasize = 0.0, 0.0
    for seed in dns_seeds
        comptime += load(
            getdatafile(outdir, params.nles[1], params.filters[1], seed),
            "comptime",
        )
    end
    for seed in dns_seeds, nles in params.nles, Φ in params.filters
        data = namedtupleload(getdatafile(outdir, nles, Φ, seed))
        datasize += Base.summarysize(data)
    end
    @info "Data" comptime
    @info "Data" comptime / 60 datasize * 1e-9
    clean()
end

# LES setups
setups = map(nles -> getsetup(; params, nles), params.nles);

########################################################################## #src

# ## CNN closure model

# All training sessions will start from the same θ₀
# for a fair comparison.

closure, θ_start, st = NS.load_model(conf)
# same model structure in INS format
closure_INS, θ_INS = cnn(;
    setup = setups[1],
    radii = [2, 2, 2, 2,2],
    channels = [24,24,24,24, 2],
    activations = [tanh,tanh,tanh,tanh, identity],
    use_bias = [true,true, true,true, false],
    rng = Xoshiro(seeds.θ_start),
)
@assert θ_start == θ_INS

@info "Initialized CNN with $(length(θ_start)) parameters"

# Give the CNN a test run
# Note: Data and parameters are stored on the CPU, and
# must be moved to the GPU before use (with `device`)
let
    @info "CNN warm up run"
    using NeuralClosure.Zygote
    u = randn(T, 32, 32, 2, 10) |> device
    θ = θ_start |> device
    closure(u, θ, st)
    gradient(θ -> sum(closure(u, θ, st)[1]), θ)
    clean()
end

########################################################################## #src

# ## Training

# ### A-priori training
#
# Train one set of CNN parameters for each of the filter types and grid sizes.
# Use the same batch selection random seed for each training setup.
# Save parameters to disk after each run.
# Plot training progress (for a validation data batch).

# Train
let
    dotrain = conf["priori"]["dotrain"]
    nepoch = conf["priori"]["nepoch"]
    dotrain && trainprior(;
        params,
        priorseed = seeds.prior,
        dns_seeds_train,
        dns_seeds_valid,
        taskid,
        outdir,
        plotdir,
        closure,
        closure_name,
        θ_start,
        st,
        opt = eval(Meta.parse(conf["priori"]["opt"])),
        batchsize = conf["priori"]["batchsize"],
        nepoch,
    )
end

# Load learned parameters and training times
priortraining = loadprior(outdir, closure_name, params.nles, params.filters)
θ_cnn_prior = map(p -> copyto!(copy(θ_start), p.θ), priortraining)
@info "" θ_cnn_prior .|> extrema # Check that parameters are within reasonable bounds

# Training times
map(p -> p.comptime, priortraining)
map(p -> p.comptime, priortraining) |> vec .|> x -> round(x; digits = 1)
map(p -> p.comptime, priortraining) |> sum |> x -> x / 60 # Minutes

# ## Plot training history

with_theme(; palette) do
    doplot() || return
    fig = Figure(; size = (950, 250))
    for (ig, nles) in enumerate(params.nles)
        ax = Axis(
            fig[1, ig];
            title = "n = $(nles)",
            xlabel = "Iteration",
            ylabel = "A-priori error",
            ylabelvisible = ig == 1,
            yticksvisible = ig == 1,
            yticklabelsvisible = ig == 1,
        )
        ylims!(-0.05, 1.05)
        lines!(
            ax,
#            [Point2f(0, 1), Point2f(priortraining[ig, 1].lhist_nomodel[end][1], 1)];
            priortraining[ig, 1].lhist_nomodel,
            label = "No closure",
            linestyle = :dash,
        )
        for (ifil, Φ) in enumerate(params.filters)
            label = Φ isa FaceAverage ? "FA" : "VA"
            lines!(ax, priortraining[ig, ifil].lhist_val; label)
        end
    end
    axes = filter(x -> x isa Axis, fig.content)
    linkaxes!(axes...)
    Legend(fig[1, end+1], axes[1])
    figdir = joinpath(plotdir, "priortraining")
    ispath(figdir) || mkpath(figdir)
    save("$figdir/validationerror.pdf", fig)
    display(fig)
end

########################################################################## #src

# ### A-posteriori training
#
# Train one set of CNN parameters for each
# projection order, filter type and grid size.
# Use the same batch selection random seed for each training setup.
# Save parameters to disk after each combination.
# Plot training progress (for a validation data batch).
#
# [INS] The time stepper `RKProject` allows for choosing when to project.
# [CNODE] Only DCF (last) is supported since it appears to be the best one.

projectorders = eval(Meta.parse(conf["posteriori"]["projectorders"]))
nprojectorders = length(projectorders)
@assert nprojectorders == 1 "Only DCF should be done"

# Train
let
    dotrain = conf["posteriori"]["dotrain"]
    nepoch = conf["posteriori"]["nepoch"]
    nepoch = 40
    dotrain && trainpost(;
        params,
        projectorders,
        outdir,
        plotdir,
        taskid,
        postseed = seeds.post,
        dns_seeds_train,
        dns_seeds_valid,
        nunroll = conf["posteriori"]["nunroll"],
        closure,
        θ_start = θ_cnn_prior,
        st,
        opt = eval(Meta.parse(conf["posteriori"]["opt"])),
        nunroll_valid = conf["posteriori"]["nunroll_valid"],
        nepoch,
        dt = eval(Meta.parse(conf["posteriori"]["dt"])),
    )
end

# Load learned parameters and training times

posttraining = loadpost(outdir, params.nles, params.filters, projectorders)
θ_cnn_post = map(p -> copyto!(copy(θ_start), p.θ), posttraining)
@info "" θ_cnn_post .|> extrema # Check that parameters are within reasonable bounds

# Training times
map(p -> p.comptime, posttraining) ./ 60
map(p -> p.comptime, posttraining) |> sum |> x -> x / 60
map(p -> p.comptime, posttraining) |> x -> reshape(x, :, nprojectorders) .|> x -> round(x; digits = 1)

# ## Plot a-posteriori training history

with_theme(; palette) do
    doplot() || return
    fig = Figure(; size = (950, 400))
    for (iorder, projectorder) in enumerate(projectorders)
        axes = []
        for (ig, nles) in enumerate(params.nles)
            ax = Axis(
                fig[iorder, ig];
                title = "n = $(nles)",
                xlabel = "Iteration",
                ylabel = projectorder == ProjectOrder.First ? "DIF" : "DCF",
                ylabelvisible = ig == 1,
                ylabelfont = :bold,
                titlevisible = iorder == 1,
                xlabelvisible = iorder == 2,
                xticksvisible = iorder == 2,
                xticklabelsvisible = iorder == 2,
            )
            for (ifil, Φ) in enumerate(params.filters)
                postfile = Benchmark.getpostfile(outdir, nles, Φ, projectorder)
                check = namedtupleload(postfile)
                # print the keys of the checkpoint
                (; lhist_val) = check[1]
                label = Φ isa FaceAverage ? "FA" : "VA"
                lines!(ax, lhist_val; color = Cycled(ifil + 1), label)
            end
            ig == 4 && iorder == 1 && axislegend(ax)
            push!(axes, ax)
        end
        # linkaxes!(axes...)
    end
    # axes = filter(x -> x isa Axis, fig.content)
    # linkaxes!(axes...)
    # Legend(fig[:, end+1], filter(x -> x isa Axis, fig.content)[1])
    Label(fig[0, :], "A-posteriori error"; valign = :bottom, font = :bold)
    rowgap!(fig.layout, 10)
    figdir = joinpath(plotdir, "posttraining")
    ispath(figdir) || mkpath(figdir)
    save("$figdir/validationerror.pdf", fig)
    display(fig)
end

########################################################################## #src

# ## Prediction errors

# ### Compute a-priori errors
#
# Note that it is still interesting to compute the a-priori errors for the
# a-posteriori trained CNN.
let
    eprior = (;
        nomodel = ones(T, length(params.nles)),
        prior = zeros(T, size(θ_cnn_prior)),
        post = zeros(T, size(θ_cnn_post)),
    )
    for (ifil, Φ) in enumerate(params.filters), (ig, nles) in enumerate(params.nles)
        @info "Computing a-priori errors" Φ nles

        setup = getsetup(; params, nles)
        data = map(s -> namedtupleload(getdatafile(outdir, nles, Φ, s)), dns_seeds_test)
        testset = create_io_arrays(data, setup)
        i = 1:min(1000, size(testset.u, 4))
        u, c = testset.u[:, :, :, i], testset.c[:, :, :, i]
        testset = (u, c) |> device
        priori_err(θ) = loss_priori_lux(closure, θ, st, testset)
        eprior.prior[ig, ifil] = priori_err(device(θ_cnn_prior[ig, ifil]))[1]
        for iorder in eachindex(projectorders)
            eprior.post[ig, ifil, iorder] = priori_err(device(θ_cnn_post[ig, ifil, iorder]))[1]
        end
    end
    jldsave(joinpath(outdir_model, "eprior.jld2"); eprior...)
end
clean()

eprior = namedtupleload(joinpath(outdir_model, "eprior.jld2"))

########################################################################## #src

# ### Compute a-posteriori errors

let
    sample = namedtupleload(
        getdatafile(outdir, params.nles[1], params.filters[1], dns_seeds_test[1]),
    )
    sample.t[100]
end

let
    s = (length(params.nles), length(params.filters), length(projectorders))
    epost = (;
        nomodel = zeros(T, s),
        cnn_prior = zeros(T, s),
        cnn_post = zeros(T, s),
    )
    for (iorder, projectorder) in enumerate(projectorders),
        (ifil, Φ) in enumerate(params.filters),
        (ig, nles) in enumerate(params.nles)

        @info "Computing a-posteriori errors" projectorder Φ nles
        I = CartesianIndex(ig, ifil, iorder)
        setup = getsetup(; params, nles)
        psolver = psolver_spectral(setup)
        sample = namedtupleload(getdatafile(outdir, nles, Φ, dns_seeds_test[1]))
        it = 1:100
        data = (;
            u = selectdim(sample.u, ndims(sample.u), it) |> collect |> device,
            t = sample.t[it],
        )
        dt = T(1e-3)
        
        ## No model
        dudt_nomod = NS.create_right_hand_side(
            setup, psolver)
        err_post = create_loss_post_lux(dudt_nomod; sciml_solver = Tsit5(), dt = dt)
        epost.nomodel[I] = err_post(closure, θ_cnn_post[I].*0 , st, data)[1]
        # with closure
        dudt = NS.create_right_hand_side_with_closure(
            setup, psolver, closure, st)
        err_post = create_loss_post_lux(dudt; sciml_solver = Tsit5(), dt = dt)
        epost.cnn_prior[I] = err_post(closure, device(θ_cnn_prior[ig, ifil]), st, data)[1]
        epost.cnn_post[I] =  err_post(closure, device(θ_cnn_post[I]), st, data)[1]
        clean()
    end
    jldsave(joinpath(outdir_model, "epost.jld2"); epost...)
end

epost = namedtupleload(joinpath(outdir_model, "epost.jld2"))


########################################################################## #src

# ### Plot a-priori errors

# Better for PDF export
CairoMakie.activate!()

with_theme(; palette) do
    fig = Figure(; size = (800, 300))
    axes = []
    for (ifil, Φ) in enumerate(params.filters)
        ax = Axis(
            fig[1, ifil];
            xscale = log10,
            xticks = params.nles,
            xlabel = "Resolution",
            # title = "Relative a-priori error $(ifil == 1 ? " (FA)" : " (VA)")",
            # title = "$(Φ isa FaceAverage ? "FA" : "VA")",
            title = "$(Φ isa FaceAverage ? "Face-averaging" : "Volume-averaging")",
            ylabel = "A-priori error",
            ylabelvisible = ifil == 1,
            yticksvisible = ifil == 1,
            yticklabelsvisible = ifil == 1,
        )
        for (e, marker, label, color) in [
            (eprior.nomodel, :circle, "No closure", Cycled(1)),
            (eprior.prior[:, ifil], :utriangle, "CNN (prior)", Cycled(2)),
            (eprior.post[:, ifil, 1], :diamond, "CNN (post, DCF)", Cycled(3)),
        ]
            scatterlines!(params.nles, e; marker, color, label)
        end
        # axislegend(; position = :lb)
        ylims!(ax, (T(-0.05), T(1.05)))
        push!(axes, ax)
    end
    Legend(fig[1, end+1], axes[1])
    save("$plotdir/eprior.pdf", fig)
    display(fig)
end

########################################################################## #src

# ### Plot a-posteriori errors

# Better for PDF export
CairoMakie.activate!()

with_theme(; palette) do
    doplot() || return
    fig = Figure(; size = (800, 300))
    linestyles = [:solid, :dash]
    linestyles = [:solid]
    for (iorder, projectorder) in enumerate(projectorders)
        lesmodel = iorder == 1 ? "DCF" : "DCF"
        (; nles) = params
        ax = Axis(
            fig[1, iorder];
            xscale = log10,
            yscale = log10,
            xticks = nles,
            xlabel = "Resolution",
            title = "$lesmodel",
            ylabel = "A-posteriori error",
            ylabelvisible = iorder == 1,
            yticksvisible = iorder == 1,
            yticklabelsvisible = iorder == 1,
        )
        for (e, marker, label, color) in [
            (epost.nomodel, :circle, "No closure", Cycled(1)),
            (epost.cnn_prior, :rect, "CNN (Lprior)", Cycled(3)),
            (epost.cnn_post, :diamond, "CNN (Lpost)", Cycled(4)),
        ]
            for (ifil, linestyle) in enumerate(linestyles)
                ifil == 2 && (label = nothing)
                scatterlines!(nles, e[:, ifil, iorder]; color, linestyle, marker, label)
            end
        end
        # ylims!(ax, (T(0.025), T(1.00)))
    end
    linkaxes!(filter(x -> x isa Axis, fig.content)...)
    g = GridLayout(fig[1, end+1])
    Legend(g[1, 1], filter(x -> x isa Axis, fig.content)[1]; valign = :bottom)
    Legend(
        g[2, 1],
        map(s -> LineElement(; color = :black, linestyle = s), linestyles),
        ["FA"];
        orientation = :horizontal,
        valign = :top,
    )
    rowsize!(g, 1, Relative(1 / 2))
    save("$plotdir/epost.pdf", fig)
    display(fig)
end

########################################################################## #src

# ## Energy evolution

# ### Compute total kinetic energy as a function of time

let
    s = length(params.nles), length(params.filters), length(projectorders)
    keys = [:ref, :nomodel, :cnn_prior, :cnn_post]
    divergencehistory = (; map(k -> k => fill(Point2f[], s), keys)...)
    energyhistory = (; map(k -> k => fill(Point2f[], s), keys)...)
    for (iorder, projectorder) in enumerate(projectorders),
        (ifil, Φ) in enumerate(params.filters),
        (ig, nles) in enumerate(params.nles)

        I = CartesianIndex(ig, ifil, iorder)
        @info "Computing divergence and kinetic energy" projectorder Φ nles
        setup = getsetup(; params, nles)
        psolver = default_psolver(setup)
        sample = namedtupleload(getdatafile(outdir, nles, Φ, dns_seeds_test[1]))
        ustart = selectdim(sample.u, ndims(sample.u), 1) |> collect |> device
        T = eltype(ustart)

        # Shorter time for DIF
        t_DIF = T(1)

        # Reference trajectories
        divergencehistory.ref[I] = let
            div = scalarfield(setup)
            udev = vectorfield(setup)
            it = 1:5:length(sample.t)
            map(it) do it
                t = sample.t[it]
                u = selectdim(sample.u, ndims(sample.u), it)
                copyto!(udev, u)
                IncompressibleNavierStokes.divergence!(div, udev, setup)
                d = view(div, setup.grid.Ip)
                d = sum(abs2, d) / length(d)
                d = sqrt(d)
                Point2f(t, d)
            end
        end
        energyhistory.ref[I] = let
            it = 1:5:length(sample.t)
            udev = vectorfield(setup)
            map(it) do it
                t = sample.t[it]
                u = selectdim(sample.u, ndims(sample.u), it)
                copyto!(udev, u)
                Point2f(t, total_kinetic_energy(udev, setup))
            end
        end

        nupdate = 5
        writer = processor() do state
            div = scalarfield(setup)
            dhist = Point2f[]
            ehist = zeros(Point2f, 0)
            on(state) do (; u, t, n)
                if n % nupdate == 0
                    IncompressibleNavierStokes.divergence!(div, u, setup)
                    d = view(div, setup.grid.Ip)
                    d = sum(abs2, d) / length(d)
                    d = sqrt(d)
                    push!(dhist, Point2f(t, d))
                    e = total_kinetic_energy(u, setup)
                    push!(ehist, Point2f(t, e))
                end
            end
            state[] = state[] # Compute initial divergence
            (; dhist, ehist)
        end

        for (sym, closure_model, θ) in [
            (:nomodel, nothing, nothing),
            (:cnn_prior, wrappedclosure(closure_INS, setup), device(θ_cnn_prior[ig, ifil])),
            (:cnn_post, wrappedclosure(closure_INS, setup), device(θ_cnn_post[I])),
        ]
            _, results = solve_unsteady(;
                setup = (; setup..., closure_model),
                ustart,
                tlims = (
                    sample.t[1],
                    projectorder == ProjectOrder.First ? t_DIF : sample.t[end],
                ),
                Δt_min = T(1e-5),
                method = RKProject(params.method, projectorder),
                processors = (; writer, logger = timelogger(; nupdate = 500)),
                psolver,
                θ,
            )
            divergencehistory[sym][I] = results.writer.dhist
            energyhistory[sym][I] = results.writer.ehist
        end
    end
    jldsave(joinpath(outdir, "history.jld2"); energyhistory, divergencehistory)
    clean()
end

(; divergencehistory, energyhistory) = namedtupleload(joinpath(outdir, "history.jld2"));

########################################################################## #src

# Check that energy is within reasonable bounds
energyhistory.ref .|> extrema
energyhistory.nomodel .|> extrema
energyhistory.cnn_prior .|> extrema
energyhistory.cnn_post .|> extrema

# Check that divergence is within reasonable bounds
divergencehistory.ref .|> extrema
divergencehistory.nomodel .|> extrema
divergencehistory.cnn_prior .|> extrema
divergencehistory.cnn_post .|> extrema

########################################################################## #src

# ### Plot energy evolution

# Better for PDF export
CairoMakie.activate!()

with_theme(; palette) do
    doplot() || return
    for (igrid, nles) in enumerate(params.nles)
        @info "Plotting energy evolution" nles
        fig = Figure(; size = (800, 450))
        g = GridLayout(fig[1, 1])
        for (iorder, projectorder) in enumerate(projectorders),
            (ifil, Φ) in enumerate(params.filters)

            I = CartesianIndex(igrid, ifil, iorder)
            subfig = g[ifil, iorder]
            ax = Axis(
                subfig;
                # xscale = log10,
                # yscale = log10,
                xlabel = "t",
                # ylabel = Φ isa FaceAverage ? "Face-average" : "Volume-average",
                ylabel = "E(t)",
                # ylabelfont = :bold,
                title = projectorder == ProjectOrder.First ? "DIF" : "DCF",
                titlevisible = ifil == 1,
                xlabelvisible = ifil == 2,
                xticksvisible = ifil == 2,
                xticklabelsvisible = ifil == 2,
                ylabelvisible = iorder == 1,
                yticksvisible = iorder == 1,
                yticklabelsvisible = iorder == 1,
            )
            # xlims!(ax, (1e-2, 5.0))
            # xlims!(ax, (0.0, 1.0))
            # ylims!(ax, (1.3, 2.3))
            plots = [
                (energyhistory.nomodel, :solid, 1, "No closure"),
                (energyhistory.cnn_prior, :solid, 3, "CNN (prior)"),
                (energyhistory.cnn_post, :solid, 4, "CNN (post)"),
                (energyhistory.ref, :dash, 1, "Reference"),
            ]
            for (p, linestyle, i, label) in plots
                lines!(ax, p[I]; color = Cycled(i), linestyle, label)
                iorder == 1 && xlims!(-0.05, 1.05)
                # iorder == 1 && ylims!(1.1, 3.1)
                ylims!(1.3, 3.0)
            end

            # Plot zoom-in box
            if iorder == 2
                tlims = iorder == 1 ? (0.05, 0.2) : (0.8, 1.2)
                i1 = findfirst(p -> p[1] > tlims[1], energyhistory.ref[I])
                i2 = findfirst(p -> p[1] > tlims[2], energyhistory.ref[I])
                tlims = energyhistory.ref[I][i1][1], energyhistory.ref[I][i2][1]
                klims = energyhistory.ref[I][i1][2], energyhistory.ref[I][i2][2]
                dk = klims[2] - klims[1]
                # klims = klims[1] - 0.2 * dk, klims[2] + 0.2 * dk
                w = iorder == 1 ? 0.2 : 0.1
                klims = klims[1] + w * dk, klims[2] - w * dk
                box = [
                    Point2f(tlims[1], klims[1]),
                    Point2f(tlims[2], klims[1]),
                    Point2f(tlims[2], klims[2]),
                    Point2f(tlims[1], klims[2]),
                    Point2f(tlims[1], klims[1]),
                ]
                lines!(ax, box; color = :black)
                ax2 = Axis(
                    subfig;
                    # bbox = BBox(0.8, 0.9, 0.2, 0.3),
                    width = Relative(0.35),
                    height = Relative(0.35),
                    halign = 0.05,
                    valign = 0.95,
                    limits = (tlims..., klims...),
                    xscale = log10,
                    yscale = log10,
                    xticksvisible = false,
                    xticklabelsvisible = false,
                    xgridvisible = false,
                    yticksvisible = false,
                    yticklabelsvisible = false,
                    ygridvisible = false,
                    backgroundcolor = :white,
                )
                # https://discourse.julialang.org/t/makie-inset-axes-and-their-drawing-order/60987/5
                translate!(ax2.scene, 0, 0, 10)
                translate!(ax2.elements[:background], 0, 0, 9)
                for (p, linestyle, i, label) in plots
                    lines!(ax2, p[igrid, ifil, iorder]; color = Cycled(i), linestyle, label)
                end
            end

            Label(
                g[ifil, 0],
                # Φ isa FaceAverage ? "Face-average" : "Volume-average";
                Φ isa FaceAverage ? "FA" : "VA";
                # halign = :right,
                font = :bold,
                # rotation = pi/2,
                tellheight = false,
            )
        end
        colgap!(g, 10)
        rowgap!(g, 10)
        # colsize!(g, 1, Relative(1 / 5))
        Legend(fig[:, end+1], filter(x -> x isa Axis, fig.content)[1])
        name = "$plotdir/energy_evolution/"
        ispath(name) || mkpath(name)
        save("$(name)/nles=$(nles).pdf", fig)
        display(fig)
    end
end

########################################################################## #src

# ### Plot Divergence

# Better for PDF export
CairoMakie.activate!()

with_theme(; palette) do
    doplot() || return
    islog = true
    for (igrid, nles) in enumerate(params.nles)
        @info "Plotting divergence" nles
        fig = Figure(; size = (800, 450))
        for (iorder, projectorder) in enumerate(projectorders),
            (ifil, Φ) in enumerate(params.filters)

            I = CartesianIndex(igrid, ifil, iorder)
            subfig = fig[ifil, iorder]
            ax = Axis(
                subfig;
                yscale = islog ? log10 : identity,
                xlabel = "t",
                ylabel = Φ isa FaceAverage ? "Face-average" : "Volume-average",
                ylabelfont = :bold,
                title = projectorder == ProjectOrder.First ? "DIF" : "DCF",
                titlevisible = ifil == 1,
                xlabelvisible = ifil == 2,
                xticksvisible = ifil == 2,
                xticklabelsvisible = ifil == 2,
                ylabelvisible = iorder == 1,
                yticksvisible = iorder == 1,
                yticklabelsvisible = iorder == 1,
            )
            lines!(ax, divergencehistory.nomodel[I]; label = "No closure")
            lines!(ax, divergencehistory.cnn_prior[I]; label = "CNN (prior)")
            lines!(ax, divergencehistory.cnn_post[I]; label = "CNN (post)")
            lines!(
                ax,
                divergencehistory.ref[I];
                color = Cycled(1),
                linestyle = :dash,
                label = "Reference",
            )
            islog && ylims!(ax, (T(1e-6), T(1e3)))
            iorder == 1 && xlims!(ax, (-0.05, 1.05))
        end
        rowgap!(fig.layout, 10)
        colgap!(fig.layout, 10)
        Legend(fig[:, end+1], filter(x -> x isa Axis, fig.content)[1])
        name = "$plotdir/divergence/"
        ispath(name) || mkpath(name)
        save("$(name)/nles=$(nles).pdf", fig)
        display(fig)
    end
end

########################################################################## #src

# ## Solutions at different times

let
    s = length(params.nles), length(params.filters), length(projectorders)
    temp = zeros(T, ntuple(Returns(0), params.D + 1))
    keys = [:ref, :nomodel, :cnn_prior, :cnn_post]
    times = T[0.1, 0.5, 1.0, 5.0]
    itime_max_DIF = 3
    times_exact = copy(times)
    utimes = map(t -> (; map(k -> k => fill(temp, s), keys)...), times)
    for (iorder, projectorder) in enumerate(projectorders),
        (ifil, Φ) in enumerate(params.filters),
        (igrid, nles) in enumerate(params.nles)

        @info "Computing test solutions" projectorder Φ nles
        I = CartesianIndex(igrid, ifil, iorder)
        setup = getsetup(; params, nles)
        psolver = default_psolver(setup)
        sample = namedtupleload(getdatafile(outdir, nles, Φ, dns_seeds_test[1]))
        ustart = selectdim(sample.u, ndims(sample.u), 1) |> collect
        t = sample.t
        solve(ustart, tlims, closure_model, θ) =
            solve_unsteady(;
                setup = (; setup..., closure_model),
                ustart = device(ustart),
                tlims,
                method = RKProject(params.method, projectorder),
                psolver,
                θ,
            )[1].u .|> Array
        t1 = t[1]
        for i in eachindex(times)
            # Only first times for First
            i > itime_max_DIF && projectorder == ProjectOrder.First && continue

            # Time interval
            t0 = t1
            t1 = times[i]

            # Adjust t1 to be exactly on a reference time
            it = findfirst(>(t1), t)
            if isnothing(it)
                # Not found: Final time
                it = length(t)
            end
            t1 = t[it]
            tlims = (t0, t1)
            times_exact[i] = t1

            getprev(i, sym) = i == 1 ? ustart : utimes[i-1][sym][I]

            # Compute fields
            utimes[i].ref[I] = selectdim(sample.u, ndims(sample.u), it) |> collect
            utimes[i].nomodel[I,:] = solve(getprev(i, :nomodel), tlims, nothing, nothing)
            utimes[i].cnn_prior[I,:] = solve(
                getprev(i, :cnn_prior),
                tlims,
                wrappedclosure(closure, setup),
                device(θ_cnn_prior[igrid, ifil]),
            )
            utimes[i].cnn_post[I,:] = solve(
                getprev(i, :cnn_post),
                tlims,
                wrappedclosure(closure, setup),
                device(θ_cnn_post[I]),
            )
        end
        clean()
    end
    jldsave("$outdir/solutions.jld2"; u = utimes, t = times_exact, itime_max_DIF)
end;

# Load solution
solutions = namedtupleload("$outdir/solutions.jld2");

########################################################################## #src

# ### Plot spectra
#
# Plot kinetic energy spectra.

with_theme(; palette) do
    doplot() || return
    for (ifil, Φ) in enumerate(params.filters), (igrid, nles) in enumerate(params.nles)
        @info "Plotting spectra" Φ nles
        fig = Figure(; size = (800, 450))
        fil = Φ isa FaceAverage ? "face-average" : "volume-average"
        setup = getsetup(; params, nles)
        (; Ip) = setup.grid
        (; inds, κ, K) = IncompressibleNavierStokes.spectral_stuff(setup)
        kmax = maximum(κ)
        allaxes = []
        for (iorder, projectorder) in enumerate(projectorders)
            rowaxes = []
            for (itime, t) in enumerate(solutions.t)
                # Only first time for First
                projectorder == ProjectOrder.First &&
                    itime > solutions.itime_max_DIF &&
                    continue

                fields = map(
                    k -> solutions.u[itime][k][igrid, ifil, iorder] |> device,
                    [:ref, :nomodel, :cnn_prior, :cnn_post],
                )
                specs = map(fields) do u
                    state = (; u)
                    spec = observespectrum(state; setup)
                    spec.ehat[]
                end
                ## Build inertial slope above energy
                logkrange = [0.45 * log(kmax), 0.85 * log(kmax)]
                krange = exp.(logkrange)
                slope, slopelabel = -T(3), L"$\kappa^{-3}$"
                slopeconst = maximum(specs[1] ./ κ .^ slope)
                offset = 3
                inertia = offset .* slopeconst .* krange .^ slope
                ## Nice ticks
                logmax = round(Int, log2(kmax + 1))
                # xticks = T(2) .^ (0:logmax)
                if logmax > 5
                    xticks = T[1, 4, 16, 64, 256]
                else
                    xticks = T[1, 2, 4, 8, 16, 32]
                end
                ## Make plot
                irow = projectorder == ProjectOrder.First ? 2 : 1
                subfig = fig[irow, itime]
                ax = Axis(
                    subfig;
                    xticks,
                    xlabel = "κ",
                    xlabelvisible = irow == 2,
                    xticksvisible = irow == 2,
                    xticklabelsvisible = irow == 2,
                    ylabel = projectorder == ProjectOrder.First ? "DIF" : "DCF",
                    ylabelfont = :bold,
                    ylabelvisible = itime == 1,
                    yticksvisible = itime == 1,
                    yticklabelsvisible = itime == 1,
                    xscale = log2,
                    yscale = log10,
                    limits = (1, kmax, T(1e-8), T(1)),
                    title = irow == 1 ? "t = $(round(t; digits = 1))" : "",
                )

                # Plot zoom-in box
                k1, k2 = klims = extrema(κ)
                center = 0.8
                dk = 0.025
                logklims = (center - dk) * log(k2), (center + dk) * log(k2)
                k1, k2 = klims = exp.(logklims)
                i1 = findfirst(>(k1), κ)
                i2 = findfirst(>(k2), κ)
                elims = specs[1][i1], specs[1][i2]
                loge1, loge2 = log.(elims)
                de = (loge1 - loge2) * 0.05
                logelims = loge1 + de, loge2 - de
                elims = exp.(logelims)
                box = [
                    Point2f(klims[1], elims[1]),
                    Point2f(klims[2], elims[1]),
                    Point2f(klims[2], elims[2]),
                    Point2f(klims[1], elims[2]),
                    Point2f(klims[1], elims[1]),
                ]
                ax_zoom = Axis(
                    subfig;
                    width = Relative(0.45),
                    height = Relative(0.4),
                    halign = 0.05,
                    valign = 0.05,
                    limits = (klims..., reverse(elims)...),
                    xscale = log10,
                    yscale = log10,
                    xticksvisible = false,
                    xticklabelsvisible = false,
                    xgridvisible = false,
                    yticksvisible = false,
                    yticklabelsvisible = false,
                    ygridvisible = false,
                    backgroundcolor = :white,
                )
                # https://discourse.julialang.org/t/makie-inset-axes-and-their-drawing-order/60987/5
                translate!(ax_zoom.scene, 0, 0, 10)
                translate!(ax_zoom.elements[:background], 0, 0, 9)

                # Plot lines in both axes
                for ax in (ax, ax_zoom)
                    lines!(ax, κ, specs[2]; color = Cycled(1), label = "No model")
                    lines!(ax, κ, specs[4]; color = Cycled(3), label = "CNN (prior)")
                    lines!(ax, κ, specs[5]; color = Cycled(4), label = "CNN (post)")
                    lines!(
                        ax,
                        κ,
                        specs[1];
                        color = Cycled(1),
                        linestyle = :dash,
                        label = "Reference",
                    )
                    lines!(
                        ax,
                        krange,
                        inertia;
                        color = Cycled(1),
                        label = slopelabel,
                        linestyle = :dot,
                    )
                end

                # Show box in main plot
                lines!(ax, box; color = :black)

                # axislegend(ax; position = :lb)
                autolimits!(ax)

                push!(allaxes, ax)
                push!(rowaxes, ax)
            end
            linkaxes!(rowaxes...)
        end
        # linkaxes!(allaxes...)
        # linkaxes!(filter(x -> x isa Axis, fig.content)...)
        Legend(
            fig[2, solutions.itime_max_DIF+1:end],
            fig.content[1];
            tellwidth = false,
            tellheight = false,
            # width = Auto(false),
            # height = Auto(false),
            # halign = :left,
            # framevisible = false,
        )
        Label(
            fig[0, 1:end],
            "Energy spectra ($fil, n = $nles)";
            valign = :bottom,
            font = :bold,
        )
        rowgap!(fig.layout, 10)
        colgap!(fig.layout, 10)
        # ylims!(ax, (T(1e-3), T(0.35)))
        specdir = "$plotdir/spectra/"
        ispath(specdir) || mkpath(specdir)
        name = splatfileparts(; filter = Φ, nles)
        save("$specdir/$name.pdf", fig)
        display(fig)
    end
end

########################################################################## #src

# ### Plot fields

# Export to PNG, otherwise each volume gets represented
# as a separate rectangle in the PDF
# (takes time to load in the article PDF)
GLMakie.activate!()

with_theme(; palette) do
    doplot() || return
    ## Reference box for eddy comparison
    x1 = 0.3
    x2 = 0.5
    y1 = 0.5
    y2 = 0.7
    box = [
        Point2f(x1, y1),
        Point2f(x2, y1),
        Point2f(x2, y2),
        Point2f(x1, y2),
        Point2f(x1, y1),
    ]
    for (ifil, Φ) in enumerate(params.filters)
        Φ isa FaceAverage || continue
        # fig = Figure(; size = (710, 400))
        fig = Figure(; size = (770, 360))
        irow = 0
        itime = 3
        for (igrid, nles) in enumerate(params.nles)
            itime == 1 && (nles in [32, 64] || continue)
            itime == 3 && (nles in [64, 128] || continue)
            # nles in [128, 256] || continue
            irow += 1
            setup = getsetup(; params, nles)
            (; Ip, xp) = setup.grid
            xplot = xp[1][2:end-1], xp[2][2:end-1]
            xplot = xplot .|> Array
            # lesmodel = iorder == 1 ? "DIF" : "DCF"
            # fig = fieldplot(
            #     (; u, temp = nothing, t = T(0));
            #     setup,
            #     title,
            #     docolorbar = false,
            #     size = (500, 500),
            # )

            utime = solutions.u[itime]
            icol = 0

            for (u, title) in [
                (utime.nomodel[igrid, ifil, 2], "No closure"),
                (utime.cnn_post[igrid, ifil, 1], "CNN (post, DIF)"),
                (utime.cnn_post[igrid, ifil, 2], "CNN (post, DCF)"),
                (utime.ref[igrid, ifil, 2], "Reference"),
            ]
                icol += 1
                u = device(u)
                w = vorticity(u, setup)
                w = interpolate_ω_p(w, setup)
                w = w[Ip] |> Array
                colorrange = get_lims(w)
                ax = Axis(
                    fig[irow, icol];
                    title,
                    xticksvisible = false,
                    xticklabelsvisible = false,
                    yticksvisible = false,
                    yticklabelsvisible = false,
                    ylabel = "n = $nles",
                    ylabelvisible = icol == 1,
                    titlevisible = irow == 1,
                    aspect = DataAspect(),
                )
                heatmap!(ax, xplot..., w; colorrange)
                lines!(ax, box; linewidth = 3, color = Cycled(2)) # Red in palette
            end
        end
        fil = Φ isa FaceAverage ? "face-average" : "volume-average"
        tlab = round(solutions.t[itime]; digits = 1)
        Label(fig[0, 1:end], "Vorticity ($fil, t = $tlab)"; valign = :bottom, font = :bold)
        colgap!(fig.layout, 10)
        rowgap!(fig.layout, 10)
        display(fig)
        path = "$plotdir/les_fields"
        ispath(path) || mkpath(path)
        name = splatfileparts(; itime, filter = Φ)
        name = joinpath(path, name)
        fname = "$(name).png"
        save(fname, fig)
    end
end

# Plot vorticity
let
    doplot() || return
    nles = 64
    sample = namedtupleload(getdatafile(outdir, nles, FaceAverage(), dns_seeds_test[1]))
    setup = getsetup(; params, nles)
    u = sample.u[1] |> device
    w = vorticity(u, setup) |> Array |> Observable
    title = sample.t[1] |> string |> Observable
    fig = heatmap(w; axis = (; title))
    for i = 1:5:1000
        u = sample.u[i] |> device
        w[] = vorticity(u, setup) |> Array
        title[] = "t = $(round(sample.t[i]; digits = 2))"
        display(fig)
        sleep(0.05)
    end
end
