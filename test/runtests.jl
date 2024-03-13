using SciMLModelCoupling
using Test

@testset "Toy example" begin
    @test helloworld() == "Hello World!"
    @test helloworld("Pablo") == "Hello Pablo!"
end

@testset "SciMLModelCoupling.jl" begin
    # Write your tests here.
end
