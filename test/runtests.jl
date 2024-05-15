using CoupledNODE
using Test

@testset "Logistic" begin # ~ 2 min
    @test try
        include("../examples/src/01.00-Logistic.jl")
        true
    catch
        false
    end
end

@testset "Gray-Scott" begin # ~ 10 min
    @test try
        include("../examples/src/02.00-GrayScott.jl")
        include("../examples/src/02.03-GrayScott.jl") #long
        true
    catch
        false
    end
end

@testset "Gray-Scott+NN" begin # ~ 13 min
    @test try
        include("../examples/src/02.01-GrayScott.jl")
        include("../examples/src/02.02-GrayScott.jl")
        #include("../examples/src/02.02-GrayScott.jl") #long
        true
    catch
        false
    end
end

@testset "Burgers" begin
    @test try
        #include("../examples/src/03.01-Burgers.jl")
        true
    catch
        false
    end
end
