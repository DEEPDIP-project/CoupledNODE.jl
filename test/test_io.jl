using CoupledNODE
using IncompressibleNavierStokes
using NeuralClosure
NS = Base.get_extension(CoupledNODE, :NavierStokes)
using Random

@testset "Read YAML" begin
    conf = NS.read_config("test_conf.yaml")
    @test conf isa Dict

    T = eval(Meta.parse(conf["T"]))
    @test T == Float32

    # test parameters
    backend = CPU()
    conf["params"]["backend"] = backend
    ref_params = (;
        D = 2,
        lims = (T(0), T(1)),
        Re = T(6e3),
        tburn = T(0.5),
        tsim = T(5),
        savefreq = 10,
        ndns = 2048,
        nles = [128],
        filters = (FaceAverage(),),
        backend,
        icfunc = (setup, psolver, rng) -> random_field(setup, T(0); kp = 20, psolver, rng),
        method = RKMethods.Wray3(; T),
        bodyforce = (dim, x, y, t) -> (dim == 1) * 5 * sinpi(8 * y),
        issteadybodyforce = true,
        processors = (; log = timelogger(; nupdate = 100)),
        Δt = T(1e-3)
    )
    params = NS.load_params(conf)
    @test params.D == ref_params.D
    @test params.lims == ref_params.lims
    @test params.Re == ref_params.Re
    @test params.tburn == ref_params.tburn
    @test params.tsim == ref_params.tsim
    @test params.savefreq == ref_params.savefreq
    @test params.ndns == ref_params.ndns
    @test params.nles == ref_params.nles
    @test params.filters == ref_params.filters
    @test params.backend == ref_params.backend
    @test params.issteadybodyforce == ref_params.issteadybodyforce
    @test params.processors == ref_params.processors
    @test params.Δt == ref_params.Δt
    # test icfunc
    setups = map(params.nles) do nles
        x = ntuple(α -> LinRange(T(0.0), T(1.0), nles + 1), params.D)
        Setup(; x = x, Re = params.Re)
    end
    setup = setups[1]
    psolver = psolver_spectral(setup)
    @test params.icfunc(setup, psolver, Xoshiro(123)) ==
          ref_params.icfunc(setup, psolver, Xoshiro(123))
    # test bodyforce 
    x = rand(1)[1]
    y = rand(1)[1]
    t = 0.0
    dim = 1
    @test params.bodyforce(dim, x, y, t) == ref_params.bodyforce(dim, x, y, t)
    # TODO: test method
    #@test params.method == ref_params.method

    # test seeds
    ref_seeds = (;
        dns = 123, # Initial conditions
        θ_start = 234, # Initial CNN parameters
        prior = 345, # A-priori training batch selection
        post = 456 # A-posteriori training batch selection
    )
    seeds = NS.load_seeds(conf)
    @test seeds.dns == ref_seeds.dns
    @test seeds.θ_start == ref_seeds.θ_start
    @test seeds.prior == ref_seeds.prior
    @test seeds.post == ref_seeds.post

    # test cnn
    ref_closure, ref_θ_start, ref_st = CoupledNODE.cnn(;
        T = T,
        D = params.D,
        data_ch = params.D,
        radii = [2, 2, 2, 2, 2],
        channels = [24, 24, 24, 24, 2],
        activations = [tanh, tanh, tanh, tanh, identity],
        use_bias = [true, true, true, true, false],
        rng = Xoshiro(seeds.θ_start)
    )
    closure, θ_start, st = NS.load_model(conf)
    #@test closure == ref_closure
    @test θ_start == ref_θ_start
    @test st == ref_st
end
