using Test
using Random: Random
using IncompressibleNavierStokes: IncompressibleNavierStokes as INS
using IncompressibleNavierStokes: default_psolver, random_field
using JLD2: load, @save
using CoupledNODE: cnn, create_loss_priori, mean_squared_error, loss_priori_lux, train
NS = Base.get_extension(CoupledNODE, :NavierStokes)
using Lux: Lux
using Optimization: Optimization
using OptimizationOptimisers: OptimizationOptimisers
using Adapt
using DifferentialEquations

T = Float32
rng = Random.Xoshiro(123)
ig = 1 # index of the LES grid to use.
params = (;
    D = 2,
    Re = T(1e3),
    lims = (T(0.0), T(1.0)),
    nles = [32],
    ndns = 64,
    filters = (FaceAverage(),),
    tburn = T(5e-2),
    tsim = T(0.2),
    savefreq = 1,
    Δt = T(5e-3), create_psolver = psolver_spectral,
    icfunc = (setup, psolver,
        rng) -> random_field(
        setup, zero(eltype(setup.grid.x[1])); kp = 20, psolver, rng),
    rng
)
d = D = params.D

@testset "Inplace vs out-of-place dynamics (CPU)" begin
    # Build LES setups and assemble operators
    setups = map(params.nles) do nles
        x = ntuple(α -> LinRange(T(0.0), T(1.0), nles + 1), params.D)
        INS.Setup(; x = x, Re = params.Re)
    end
    setup = setups[ig]

    # Initial conditions
    psolver = default_psolver(setup)
    ustart = random_field(setup)

    # The two rhs to compare
    Fout = INS.create_right_hand_side(setup, psolver)
    Fin = NS.create_right_hand_side_inplace(setup, psolver)

    # Now use SciML to solve the DNS
    tspan = (T(0), T(0.5))
    prob_in = ODEProblem(Fin, ustart, tspan, nothing)
    prob_out = ODEProblem(Fout, ustart, tspan, nothing)

    # First a burn-in simulation
    burn_in = solve(prob_in, Tsit5(); u0 = ustart, p = nothing,
        adaptive = true, saveat = 0.5, dt = T(1e-5), tspan = tspan)
    burn_out = solve(prob_out, Tsit5(); u0 = ustart, p = nothing,
        adaptive = true, saveat = 0.5, dt = T(1e-5), tspan = tspan)
    @test burn_in.u ≈ burn_out.u
    ustart = burn_in.u[end]

    # Dry runs to warm up the cache
    _, _,
    _,
    _ = @timed solve(
        prob_out, Tsit5(); u0 = ustart, p = nothing,
        adaptive = true, saveat = T(1e-4), dt = T(1e-5), tspan = tspan)
    _, _,
    _,
    _ = @timed solve(
        prob_in, Tsit5(); u0 = ustart, p = nothing,
        adaptive = true, saveat = T(1e-4), dt = T(1e-5), tspan = tspan)

    # Now the real simulations
    solution_in, t_in,
    mem_in,
    _ = @timed solve(
        prob_in, Tsit5(); u0 = ustart, p = nothing,
        adaptive = true, saveat = T(1e-4), dt = T(1e-5), tspan = tspan)
    solution_out, t_out,
    mem_out,
    _ = @timed solve(
        prob_out, Tsit5(); u0 = ustart, p = nothing,
        adaptive = true, saveat = T(1e-4), dt = T(1e-5), tspan = tspan)

    @test solution_in.u ≈ solution_out.u
    @test solution_in.t ≈ solution_out.t
    @test mem_in < mem_out
    @test t_in < t_out
end

@testset "Inplace vs out-of-place dynamics (GPU)" begin
    if !CUDA.functional()
        @testset "CUDA not available" begin
            @test true
        end
        return
    end

    # Use gpu device
    backend = CUDABackend()
    CUDA.allowscalar(false)
    device = x -> adapt(CuArray, x)

    # Build LES setups and assemble operators
    setups = map(params.nles) do nles
        x = ntuple(α -> LinRange(T(0.0), T(1.0), nles + 1), params.D)
        INS.Setup(; x = x, Re = params.Re, backend = backend)
    end
    setup = setups[ig]

    # Initial conditions
    psolver = default_psolver(setup)
    ustart = random_field(setup)

    # The two rhs to compare
    Fout = INS.create_right_hand_side(setup, psolver)
    Fin = NS.create_right_hand_side_inplace(setup, psolver)

    # Now use SciML to solve the DNS
    tspan = (T(0), T(0.5))
    prob_in = ODEProblem(Fin, ustart, tspan, nothing)
    prob_out = ODEProblem(Fout, ustart, tspan, nothing)

    # First a burn-in simulation
    burn_in = solve(prob_in, Tsit5(); u0 = ustart, p = nothing,
        adaptive = true, saveat = 0.5, dt = T(1e-5), tspan = tspan)
    burn_out = solve(prob_out, Tsit5(); u0 = ustart, p = nothing,
        adaptive = true, saveat = 0.5, dt = T(1e-5), tspan = tspan)
    @test burn_in.u ≈ burn_out.u
    @test is_on_gpu(burn_in.u[end])
    @test is_on_gpu(burn_out.u[end])
    ustart = burn_in.u[end]

    # Dry runs to warm up the cache
    _, _,
    _,
    _ = @timed solve(
        prob_out, Tsit5(); u0 = ustart, p = nothing,
        adaptive = true, saveat = T(1e-4), dt = T(1e-5), tspan = tspan)
    _, _,
    _,
    _ = @timed solve(
        prob_in, Tsit5(); u0 = ustart, p = nothing,
        adaptive = true, saveat = T(1e-4), dt = T(1e-5), tspan = tspan)

    # Now the real simulations
    solution_in, t_in,
    mem_in,
    _ = @timed solve(
        prob_in, Tsit5(); u0 = ustart, p = nothing,
        adaptive = true, saveat = T(1e-4), dt = T(1e-5), tspan = tspan)
    solution_out, t_out,
    mem_out,
    _ = @timed solve(
        prob_out, Tsit5(); u0 = ustart, p = nothing,
        adaptive = true, saveat = T(1e-4), dt = T(1e-5), tspan = tspan)

    @test solution_in.u ≈ solution_out.u
    @test solution_in.t ≈ solution_out.t
    @test mem_in < mem_out
    @test t_in < t_out
    @test is_on_gpu(solution_in.u[end])
    @test is_on_gpu(solution_out.u[end])
end

@testset "With Closure (CPU)" begin
    # Build LES setups and assemble operators
    setups = map(params.nles) do nles
        x = ntuple(α -> LinRange(T(0.0), T(1.0), nles + 1), params.D)
        INS.Setup(; x = x, Re = params.Re)
    end
    setup = setups[ig]

    # Initial conditions
    psolver = default_psolver(setup)
    ustart = random_field(setup)

    # Creation of the model: NN closure
    closure, θ,
    st = cnn(;
        T = T,
        D = D,
        data_ch = D,
        radii = [3, 3],
        channels = [2, 2],
        activations = [tanh, identity],
        use_bias = [false, false],
        rng
    )

    # The two rhs to compare
    Fout = NS.create_right_hand_side_with_closure(setup, psolver, closure, st)
    Fin = NS.create_right_hand_side_with_closure_inplace(setup, psolver, closure, st)

    # Now use SciML to solve the DNS
    tspan = (T(0), T(0.5))
    prob_in = ODEProblem(Fin, ustart, tspan, θ)
    prob_out = ODEProblem(Fout, ustart, tspan, θ)

    # First a burn-in simulation
    burn_in = solve(prob_in, Tsit5(); u0 = ustart, p = θ,
        adaptive = true, saveat = T(0.5), dt = T(1e-5), tspan = tspan)
    burn_out = solve(prob_out, Tsit5(); u0 = ustart, p = θ,
        adaptive = true, saveat = T(0.5), dt = T(1e-5), tspan = tspan)
    @test burn_in.u ≈ burn_out.u
    ustart = burn_in.u[end]

    # Dry runs to warm up the cache
    _, _,
    _,
    _ = @timed solve(
        prob_out, Tsit5(); u0 = ustart, p = θ,
        adaptive = true, saveat = T(1e-4), dt = T(1e-5), tspan = tspan)
    _, _,
    _,
    _ = @timed solve(
        prob_in, Tsit5(); u0 = ustart, p = θ,
        adaptive = true, saveat = T(1e-4), dt = T(1e-5), tspan = tspan)

    # Now the real simulations
    solution_in, t_in,
    mem_in,
    _ = @timed solve(
        prob_in, Tsit5(); u0 = ustart, p = θ,
        adaptive = true, saveat = T(1e-4), dt = T(1e-5), tspan = tspan)
    solution_out, t_out,
    mem_out,
    _ = @timed solve(
        prob_out, Tsit5(); u0 = ustart, p = θ,
        adaptive = true, saveat = T(1e-4), dt = T(1e-5), tspan = tspan)

    @test solution_in.u ≈ solution_out.u
    @test solution_in.t ≈ solution_out.t
    @test mem_in < mem_out
    @test t_in < t_out
end

@testset "With Closure (GPU)" begin
    if !CUDA.functional()
        @testset "CUDA not available" begin
            @test true
        end
        return
    end

    # Use gpu device
    backend = CUDABackend()
    CUDA.allowscalar(false)
    device = x -> adapt(CuArray, x)

    # Build LES setups and assemble operators
    setups = map(params.nles) do nles
        x = ntuple(α -> LinRange(T(0.0), T(1.0), nles + 1), params.D)
        INS.Setup(; x = x, Re = params.Re, backend = backend)
    end
    setup = setups[ig]

    # Initial conditions
    psolver = default_psolver(setup)
    ustart = random_field(setup)

    # Creation of the model: NN closure
    closure, θ,
    st = cnn(;
        T = T,
        D = D,
        data_ch = D,
        radii = [3, 3],
        channels = [2, 2],
        activations = [tanh, identity],
        use_bias = [false, false],
        rng,
        use_cuda = true
    )

    # The two rhs to compare
    Fout = NS.create_right_hand_side_with_closure(setup, psolver, closure, st)
    Fin = NS.create_right_hand_side_with_closure_inplace(setup, psolver, closure, st)

    # Now use SciML to solve the DNS
    tspan = (T(0), T(0.5))
    prob_in = ODEProblem(Fin, ustart, tspan, θ)
    prob_out = ODEProblem(Fout, ustart, tspan, θ)

    # First a burn-in simulation
    burn_in = solve(prob_in, Tsit5(); u0 = ustart, p = θ,
        adaptive = true, saveat = 0.5, dt = T(1e-5), tspan = tspan)
    burn_out = solve(prob_out, Tsit5(); u0 = ustart, p = θ,
        adaptive = true, saveat = 0.5, dt = T(1e-5), tspan = tspan)
    @test burn_in.u ≈ burn_out.u
    @test is_on_gpu(burn_in.u[end])
    @test is_on_gpu(burn_out.u[end])
    ustart = burn_in.u[end]

    # Dry runs to warm up the cache
    _, _,
    _,
    _ = @timed solve(
        prob_out, Tsit5(); u0 = ustart, p = θ,
        adaptive = true, saveat = T(1e-4), dt = T(1e-5), tspan = tspan)
    _, _,
    _,
    _ = @timed solve(
        prob_in, Tsit5(); u0 = ustart, p = θ,
        adaptive = true, saveat = T(1e-4), dt = T(1e-5), tspan = tspan)

    # Now the real simulations
    solution_in, t_in,
    mem_in,
    _ = @timed solve(
        prob_in, Tsit5(); u0 = ustart, p = θ,
        adaptive = true, saveat = T(1e-4), dt = T(1e-5), tspan = tspan)
    solution_out, t_out,
    mem_out,
    _ = @timed solve(
        prob_out, Tsit5(); u0 = ustart, p = θ,
        adaptive = true, saveat = T(1e-4), dt = T(1e-5), tspan = tspan)

    @test solution_in.u ≈ solution_out.u
    @test solution_in.t ≈ solution_out.t
    @test mem_in < mem_out
    @test t_in < t_out
    @test is_on_gpu(solution_in.u[end])
    @test is_on_gpu(solution_out.u[end])
end
