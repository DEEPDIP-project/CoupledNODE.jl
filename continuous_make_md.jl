# You can run this script to have your markdown files updated every time you modify and example.
# Particularly useful to see the rendering of Latex equations in the markdown file.

using Pkg
Pkg.add("Literate")
using Literate
using FileWatching
using Glob

cd("examples/src")

files = glob("*jl")

while true
    for file in files
        Literate.markdown(file;
            flavor = Literate.CommonMarkFlavor(),
            codefence = "```julia" => "```")
    end

    #watch_file("Example1.jl")
    watch_folder("examples/src")
end
