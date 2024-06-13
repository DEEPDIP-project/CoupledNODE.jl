function f(; x = 1, kwargs...)
    return x * x
end

f(4)
f(; x = 4)
function j(; x = 1, kwargs...)
    println(kwargs)
    return f(kwargs...)
end
j(; x = 5)
