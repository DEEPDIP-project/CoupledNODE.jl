using Literate
using Glob

cd("examples")

files = glob("Example*")

for f in files
    Literate.notebook(f ; execute = false)
    Literate.markdown(
        f ;
        flavor = Literate.CommonMarkFlavor(),
        codefence = "```julia" => "```",
    )
end