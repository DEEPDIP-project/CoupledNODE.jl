using Literate
using Glob

cd("examples")

files = glob("Example*jl")

for f in files
    println("MD for $(f)")
    Literate.markdown(
        f ;
        flavor = Literate.CommonMarkFlavor(),
        codefence = "```julia" => "```",
    )
    println("NB for $(f)")
    Literate.notebook(f ; execute = false)
end