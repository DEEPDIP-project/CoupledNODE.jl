using Test
using CoupledNODE

@testset "CUDA" begin
    # TODO: this test is broken and it breaks the CI so it is skipped
    using Pkg
    Pkg.add("CUDA")
    using CUDA
    Pkg.add("CUDSS")
    using CUDSS
    Pkg.add("cuDNN")
    using cuDNN
    Pkg.add("LuxCUDA")
    using LuxCUDA
    Cuda_ext = Base.get_extension(CoupledNODE, :CoupledNODECUDA)
    ArrayType = Cuda_ext.ArrayType()
    @test ArrayType == CUDA.CuArray || ArrayType == Array
    @test Cuda_ext.allowscalar(false) == nothing
    @test Cuda_ext.allowscalar(true) == nothing
end
