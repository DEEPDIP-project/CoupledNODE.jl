using DifferentialEquations
using IncompressibleNavierStokes: right_hand_side!, apply_bc_u!, momentum!, project!

function create_les_data_projected(;
        D,
        Re,
        lims,
        nles,
        ndns,
        filters,
        tburn,
        tsim,
        savefreq,
        Δt = nothing,
        method = RKMethods.RK44(; T = typeof(Re)),
        create_psolver = default_psolver,
        backend = IncompressibleNavierStokes.CPU(),
        icfunc = (setup, psolver, rng) -> random_field(setup, typeof(Re)(0); psolver, rng),
        processors = (; log = timelogger(; nupdate = 10)),
        rng,
        filenames = nothing,
        kwargs...
)
    @assert length(nles) == 1 "We only support one les at a time"
    @assert length(filters) == 1 "We only support one filter at a time"

    T = typeof(Re)

    compression = div.(ndns, nles)
    @assert all(==(ndns), compression .* nles)

    # Build setup and assemble operators
    dns = Setup(; x = ntuple(α -> LinRange(lims..., ndns + 1), D), Re, backend, kwargs...)
    les = map(
        nles -> Setup(;
            x = ntuple(α -> LinRange(lims..., nles + 1), D),
            Re,
            backend,
            kwargs...
        ),
        nles
    )

    # Since the grid is uniform and identical for x and y, we may use a specialized
    # spectral pressure solver
    psolver = create_psolver(dns)
    psolver_les = create_psolver.(les)

    # Initial conditions
    ustart = icfunc(dns, psolver, rng)

    any(u -> any(isnan, u), ustart) && @warn "Initial conditions contain NaNs"

    _dns = dns
    _les = les

    # Solve burn-in DNS using INS
    (; u, t),
    outputs = solve_unsteady(; setup = _dns, ustart, tlims = (T(0), tburn), Δt, psolver)
    @info "Burn-in DNS simulation finished"

    # Define the callback function for the filter
    Φ = filters[1]
    compression = compression[1]
    les = les[1]
    psolver_les = psolver_les[1]
    tdatapoint = collect(T(0):(savefreq * Δt):tsim)
    function condition(u, t, integrator)
        t in tdatapoint && return true
        return false
    end
    all_ules = Array{T}(undef, (nles[1] + 2, nles[1]+2, D, length(tdatapoint)-1))
    all_c = Array{T}(undef, (nles[1]+2, nles[1]+2, D, length(tdatapoint)-1))
    all_t = Array{T}(undef, (length(tdatapoint)-1))
    idx = Ref(1)
    u0 = copy(u)
    Fdns = INS.create_right_hand_side(dns, psolver)
    p = scalarfield(les)
    Φu = vectorfield(les)
    FΦ = vectorfield(les)
    ΦF = vectorfield(les)
    c = vectorfield(les)
    temp = nothing
    F = Fdns(u, nothing, T(0)) #TODO check if we can avoid recomputing this
    ut = copy(u)
    function filter_callback(integrator)
        GC.gc()
        CUDA.reclaim()
        ut .= integrator.u
        t = integrator.t
        F .= Fdns(u, nothing, t) #TODO check if we can avoid recomputing this

        Φ(Φu, ut, les, compression)
        apply_bc_u!(Φu, t, les)
        Φ(ΦF, F, les, compression)
        momentum!(FΦ, Φu, temp, t, les)
        apply_bc_u!(FΦ, t, les; dudt = true)
        project!(FΦ, les; psolver = psolver_les, p = p)
        @. c = ΦF - FΦ

        all_ules[:, :, :, idx[]] .= Array(Φu)
        all_c[:, :, :, idx[]] .= Array(c)
        all_t[idx[]] = t
        idx[] += 1
    end
    cb = DiscreteCallback(condition, filter_callback)

    # Now use SciML to solve the DNS
    rhs! = create_right_hand_side_inplace(dns, psolver)
    #tspan = (T(0), tsim)
    #prob = ODEProblem(rhs!, u, tspan, nothing)
    #@info "Starting DNS simulation"
    #dns_solution = solve(
    #    prob, RK4(); u0 = u, p = nothing,
    #    adaptive = false, dt=T(1e-4), saveat = 2*tsim, callback = cb, tspan = tspan, tstops = tdatapoint)
    #@info "DNS simulation finished"
    #res1 = (; u = all_ules, c = all_c, t = all_t)

    all_ules = Array{T}(undef, (nles[1] + 2, nles[1]+2, D, length(tdatapoint)-1))
    all_c = Array{T}(undef, (nles[1]+2, nles[1]+2, D, length(tdatapoint)-1))
    all_t = Array{T}(undef, (length(tdatapoint)-1))
    idx = Ref(1)

    # Setup
    t0 = T(0)
    tfinal = tsim
    nchunks = 10
    dt_chunk = tsim / nchunks
    save_every = 2 * tsim
    tchunk = collect(t0:dt_chunk:tfinal)  # Save at the end of each chunk

    u_current = u0  # Initial condition

    @info "Starting chunked DNS simulation"

    for (i, t_start) in enumerate(tchunk[1:(end - 1)])
        @info "Processing chunk $(i) from $(t_start) to $(tchunk[i+1])"
        @info size(u_current)
        @info sum(u_current)
        GC.gc()
        CUDA.reclaim()
        t_end = tchunk[i + 1]
        tspan_chunk = (t_start, t_end)
        prob = ODEProblem(rhs!, u_current, tspan_chunk, nothing)

        sol = solve(
            prob, RK4(); u0 = u_current, p = nothing,
            adaptive = false, dt = Δt, save_end = true, callback = cb,
            tspan = tspan_chunk, tstops = tdatapoint
        )

        u_current = sol.u[end]  # Use last state as initial condition for next chunk
    end

    @info "DNS simulation finished"

    res2 = (; u = all_ules, c = all_c, t = all_t)

    #@info "Comparing results"
    #@assert length(res1.t) == length(res2.t) "Time vectors have different lengths"
    #@assert size(res1.u) == size(res2.u) "Velocity fields have different sizes"
    #@assert size(res1.c) == size(res2.c) "Closure fields have different sizes"
    #for i in 1:length(res1.t)
    #    @assert res1.t[i] ≈ res2.t[i] "Time vectors differ at t = $(res1.t[i])"
    #    @assert all(isapprox.(res1.u[:, :, :, i], res2.u[:, :, :, i]; atol = 1e-4)) "Velocity fields differ at t = $(res1.t[i])"
    #    @assert all(isapprox.(res1.c[:, :, :, i], res2.c[:, :, :, i]; atol = 1e-4)) "Closure fields differ at t = $(res1.t[i])"
    #end
    #@info "Data creation finished"

    (; u = all_ules, c = all_c, t = all_t)
end
