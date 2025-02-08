using Lux, ComponentArrays, CUDA

# Define the convolutional layer
conv_layer = Conv((3, 3), 1 => 16, relu; use_bias = true)

# Create a chain with the convolutional layer
chain = Chain(conv_layer)

# Initialize parameters and state
rng = Random.default_rng()
params, state = Lux.setup(rng, chain)

# Transfer parameters and state to the GPU
params = ComponentArray(params) |> gpu
state = state |> gpu

# Define a function to apply the chain
function apply_chain(x, params, state)
    return chain(x, params, state)
end

# Example input
input_data = rand(Float32, 32, 32, 1, 1) |> gpu

# Apply the chain
output = apply_chain(input_data, params, state)

