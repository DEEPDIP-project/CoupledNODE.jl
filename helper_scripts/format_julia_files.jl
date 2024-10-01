using JuliaFormatter: format

for file in ARGS
    format(file)
    println("Formatted $file")
end
