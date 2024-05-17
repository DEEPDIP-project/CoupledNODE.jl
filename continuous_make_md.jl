# You can run this script to have your markdown files updated every time you modify and example.
# Particularly useful to see the rendering of Latex equations in the markdown file.
#
# Remember to execute this in a separate REPL started where .git is

using Pkg
Pkg.add("Literate")
using Literate
using FileWatching
using Glob

cd("examples/")

files = glob("src/*jl")

while true
    for file in files
        Literate.markdown(file;
            flavor = Literate.CommonMarkFlavor(),
            codefence = "```julia" => "```")
    end
    watch_folder("src")
end
