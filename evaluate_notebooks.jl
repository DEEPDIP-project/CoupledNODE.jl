using Literate
using Glob

cd("examples")

files = glob("*-*.jl")

for f in files
    if occursin("Logistic", f)
        Literate.notebook(f; execute = true)
    end
end
