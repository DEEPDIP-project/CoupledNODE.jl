using DifferentialEquations
using JLD2: jldsave
using Random: Random
using IncompressibleNavierStokes
using NeuralClosure
NS = Base.get_extension(CoupledNODE, :NavierStokes)

if CUDA.functional()
    backend = CUDABackend()
else
    backend = IncompressibleNavierStokes.CPU()
end

T = Float32
rng = Random.Xoshiro(123)

# Parameters
params = (;
    D = 2,
    Re = T(1e3),
    lims = (T(0.0), T(1.0)),
    nles = [16],
    ndns = 64,
    filters = (FaceAverage(),),
    tburn = T(0.1),
    tsim = T(0.5),
    savefreq = 100,
    Δt = T(1e-4),
    create_psolver = psolver_spectral,
    icfunc = (setup, psolver,
        rng) -> random_field(
        setup, zero(eltype(setup.grid.x[1])); kp = 20, psolver, rng),
    rng
)

# Reference function to test if the chunking procedure used by CoupledNODE
# works correctly for the projected LES data.
function _create_les_data_projected_singlerun(;
        u0,
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
    ustart = u0

    # We skip the burn-in phase here, since we assume that the initial conditions is right after the burn-in phase
    u = u0

    any(u -> any(isnan, u), ustart) && @warn "Initial conditions contain NaNs"

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
    Fdns = IncompressibleNavierStokes.create_right_hand_side(dns, psolver)
    p = scalarfield(les)
    Φu = vectorfield(les)
    FΦ = vectorfield(les)
    ΦF = vectorfield(les)
    c = vectorfield(les)
    temp = nothing
    F = Fdns(u, nothing, T(0)) #TODO check if we can avoid recomputing this
    ut = copy(u)
    function filter_callback(integrator)
        ut .= integrator.u
        t = integrator.t
        F .= Fdns(ut, nothing, t) #TODO check if we can avoid recomputing this

        Φ(Φu, ut, les, compression)
        IncompressibleNavierStokes.apply_bc_u!(Φu, t, les)
        Φ(ΦF, F, les, compression)
        IncompressibleNavierStokes.momentum!(FΦ, Φu, temp, t, les)
        IncompressibleNavierStokes.apply_bc_u!(FΦ, t, les; dudt = true)
        IncompressibleNavierStokes.project!(FΦ, les; psolver = psolver_les, p = p)
        @. c = ΦF - FΦ

        all_ules[:, :, :, idx[]] .= Array(Φu)
        all_c[:, :, :, idx[]] .= Array(c)
        all_t[idx[]] = t
        idx[] += 1
    end
    cb = DiscreteCallback(condition, filter_callback)

    # Now use SciML to solve the DNS
    rhs! = NS.create_right_hand_side_inplace(dns, psolver)
    tspan = (T(0), tsim)
    prob = ODEProblem(rhs!, u, tspan, nothing)
    dns_solution = solve(
        prob, Tsit5(); u0 = u, p = nothing,
        adaptive = true, saveat = 2*tsim, callback = cb, tspan = tspan, tstops = tdatapoint)

    (; u = all_ules, c = all_c, t = all_t)
end

@testset "Create data (chunking)" begin
    data = NS.create_les_data_projected(nchunks = 7;
        params...,
        backend = backend
    )
    data_ref = _create_les_data_projected_singlerun(;
        data.u0,
        params...,
        backend = backend
    )
    @test length(data.t) == length(data_ref.t)
    @test size(data.u) == size(data_ref.u)
    @test size(data.c) == size(data_ref.c)
    for i in 1:length(data.t)
        @test data.t[i] == data_ref.t[i]
        @test all(isapprox.(data.u[:, :, :, i], data_ref.u[:, :, :, i]; atol = 1e-4))
        @test all(isapprox.(data.c[:, :, :, i], data_ref.c[:, :, :, i]; atol = 1e-4))
    end
end
