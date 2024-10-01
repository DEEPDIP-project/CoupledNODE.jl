using Literate: Literate
using Glob: glob

cd("examples")

# if this is called with an argument, use that as filenames
if length(ARGS) > 0
    files = ARGS
    # only check files in examples/ and remove the path
    files = [f for f in files if occursin("examples/", f)]
    files = [replace(f, r"examples/" => "") for f in files]
else
    files = glob("src/*-*.jl")
end

overwrite_nb = true
autorun_notebooks = false

for f in files
    Literate.markdown(f;
        flavor = Literate.CommonMarkFlavor(),
        codefence = "```julia" => "```")
    if overwrite_nb
        Literate.notebook(f; execute = autorun_notebooks)
    else
        # Check if the notebook exists and if not, create it
        if !isfile(replace(f, r"\.jl$" => ".ipynb"))
            Literate.notebook(f; execute = autorun_notebooks)
        end
    end
end
