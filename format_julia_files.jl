using JuliaFormatter

for file in ARGS
    format(file)
    println("Formatted $file")
end
