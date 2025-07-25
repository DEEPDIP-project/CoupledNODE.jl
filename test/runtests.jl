using CoupledNODE
using Test
using CUDA
using Random

# Needs to be run before tests
@testset "generate test data" begin
    include("generate_test_data.jl")
end

# Check if a GPU device is available
gpu_available = CUDA.functional()
if gpu_available
    @warn "GPU device available"
else
    @warn "No GPU device available"
end

# Helper function to check if a variable is on the GPU
function is_on_gpu(x)
    return x isa CuArray || (x isa SubArray && is_on_gpu(x.parent))
end

#=
Don't add your tests to runtests.jl. Instead, create files named

    test-title-for-my-test.jl

The file will be automatically included inside a `@testset` with title "Title For My Test".
=#
for (root, dirs, files) in walkdir(@__DIR__)
    for file in files
        if isnothing(match(r"^test_.*\.jl$", file))
            continue
        end

        title = titlecase(replace(splitext(file[6:end])[1], "-" => " "))
        @testset "$title" begin
            include(file)
        end
    end
end
