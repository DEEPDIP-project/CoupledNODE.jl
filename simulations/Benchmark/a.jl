using CUDA

dt = t[2] - t[1]
tspan = [t[1], t[end]]

dt = @views t[2:2] .- t[1:1]  # Slice and subtract
dt = dt[1]  # Extract the scalar result
