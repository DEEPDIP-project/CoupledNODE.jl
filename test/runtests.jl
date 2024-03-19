using CoupledNODE
using Test

@testset "Toy example" begin
    @test helloworld() == "Hello World!"
    @test helloworld("Pablo") == "Hello Pablo!"
end

@testset "CoupledNODE.jl" begin
    # Write your tests here.
end
