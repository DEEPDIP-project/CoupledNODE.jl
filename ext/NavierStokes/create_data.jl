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
    tsave = collect(T(0):(savefreq * Δt):tsim)
    function condition(u, t, integrator)
        t in tsave && return true
        return false
    end
    all_ules = []
    all_c = []
    all_t = []
    Fdns = create_right_hand_side(dns, psolver)
    function filter_callback(integrator)
        u = integrator.u
        t = integrator.t
        F = Fdns(u, nothing, t) #TODO check if we can avoid recomputing this
        p = scalarfield(les)
        Φu = vectorfield(les)
        FΦ = vectorfield(les)
        ΦF = vectorfield(les)
        c = vectorfield(les)
        temp = nothing

        Φ(Φu, u, les, compression)
        apply_bc_u!(Φu, t, les)
        Φ(ΦF, F, les, compression)
        momentum!(FΦ, Φu, temp, t, les)
        apply_bc_u!(FΦ, t, les; dudt = true)
        project!(FΦ, les; psolver = psolver_les, p = p)
        @. c = ΦF - FΦ
        push!(all_ules, Array(Φu))
        push!(all_c, Array(c))
        push!(all_t, T(t))
    end
    cb = DiscreteCallback(condition, filter_callback)

    # Now use SciML to solve the DNS
    rhs!(du, u, p, t) = right_hand_side!(du, u, Ref([dns, psolver]), t)
    tspan = (T(0), tsim)
    prob = ODEProblem(rhs!, u, tspan, nothing)
    dns_solution = solve(
        prob, Tsit5(); u0 = u, p = nothing,
        adaptive = true, saveat = tsim, callback = cb, tspan = tspan, tstops = tsave)

    @info "DNS simulation finished"

    (; u = all_ules, c = all_c, t = all_t)
end
