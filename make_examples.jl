using Literate
using Glob

cd("examples")

files = glob("Example*jl")
overwrite_nb = false
autorun_notebooks = false

for f in files
    Literate.markdown(
        f ;
        flavor = Literate.CommonMarkFlavor(),
        codefence = "```julia" => "```",
    )
    if overwrite_nb
        Literate.notebook(f ; execute = autorun_notebooks)
    else
        # Check if the notebook exists and if not, create it
        if !isfile(replace(f, r"\.jl$" => ".ipynb"))
            Literate.notebook(f ; execute = autorun_notebooks)
        end 
    end
end