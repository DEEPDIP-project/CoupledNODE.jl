using Test
using CoupledNODE

@testset "CUDA" begin
    using Pkg
    Pkg.add("CUDA")
    using CUDA
    Pkg.add("LuxCUDA")
    using LuxCUDA
    Cuda_ext = Base.get_extension(CoupledNODE, :CoupledNODECUDA)
    ArrayType = Cuda_ext.ArrayType()
    @test ArrayType == CUDA.CuArray || ArrayType == Array
    @test Cuda_ext.allowscalar(false) == nothing
    @test Cuda_ext.allowscalar(true) == nothing
end
