using Lux: Lux, relu
using LuxCore: AbstractLuxLayer

# Observation: the original CNO paper basically assumes PBC everywhere.
# In the current implementation, we assume PBC in the upsample and downsample layers, but for the convolution kernels we use Lux which does NOT assume PBC. Think about this and decide if you want to change it.

# expected input shape: [N, N, d, batch]
# expected output shape: [N, N, d, batch]

struct CNODownsampleBlock
    T::Type
    N::Int
    D::Int
    ch_in::Int
    ch_out::Int
    k_radius::Int
    use_bias::Bool
    use_batchnorm::Bool
    layers::Tuple
end

function CNODownsampleBlock(
        N::Int,
        D::Int,
        down_factor::Int,
        ch_in::Int,
        ch_out::Int
        ;
        k_radius = 3,
        use_bias = false,
        use_batchnorm = true,
        cutoff = 0.1,        
        T = Float32,
        activation_function = relu,
        init_weight = Lux.glorot_uniform,
        )
    downsampler = create_CNOdownsampler(T, D, N, down_factor, cutoff)
    activation = create_CNOactivation(T, D, N, cutoff, activation_function = activation_function)

    layers = (
        Lux.Conv(
             ntuple(α -> 2*k_radius + 1, D),
             ch_in => ch_out,
             identity;
             use_bias = use_bias,
             init_weight = init_weight,
             pad = (ntuple(α -> 2*k_radius + 1, D) .- 1) .÷ 2
        ),
        Lux.WrappedFunction(activation),
        if use_batchnorm
            Lux.WrappedFunction(downsampler),
            Lux.BatchNorm(ch_out)
        else
            Lux.WrappedFunction(downsampler)
        end
    )

    CNODownsampleBlock(T, N, D, ch_in, ch_out, k_radius, use_bias, use_batchnorm, layers)
end


struct CNOUpsampleBlock
    T::Type
    N::Int
    D::Int
    ch_in::Int
    ch_out::Int
    k_radius::Int
    use_bias::Bool
    use_batchnorm::Bool
    layers::Tuple
end

function CNOUpsampleBlock(
        N::Int,
        D::Int,
        up_factor::Int,
        ch_in::Int,
        ch_out::Int
        ;
        k_radius = 3,
        use_bias = false,
        use_batchnorm = true,
        cutoff = 0.1,        
        T = Float32,
        activation_function = relu,
        init_weight = Lux.glorot_uniform,
        )
    upsampler = create_CNOupsampler(T, D, N, up_factor, cutoff)
    activation = create_CNOactivation(T, D, N, cutoff, activation_function = activation_function)

    layers = (
        Lux.Conv(
             ntuple(α -> 2*k_radius + 1, D),
             ch_in => ch_out,
             identity;
             use_bias = use_bias,
             init_weight = init_weight,
             pad = (ntuple(α -> 2*k_radius + 1, D) .- 1) .÷ 2
        ),
        Lux.WrappedFunction(activation),
        if use_batchnorm
            Lux.WrappedFunction(upsampler),
            Lux.BatchNorm(ch_out)
        else
            Lux.WrappedFunction(upsampler)
        end
    )

    CNOUpsampleBlock(T, N, D, ch_in, ch_out, k_radius, use_bias, use_batchnorm, layers)
end


function create_CNO(T, N, D; channels, down_factors, up_factors, cutoffs, use_batchnorms, use_biases, conv_per_block)
    layers = ()
    for i in 1:length(channels)-1 
        layers = (layers..., CNODownsampleBlock(N, D, down_factors[i], channels[i], channels[i+1], use_batchnorm = use_batchnorms[i], use_bias = use_biases[i]).layers)
    end
    return layers

    here you see the problem: if your blocks return the layers you can not easily apply them in the UNet fashion, because you have problem to 
    1) store the parameters of all the block
    2) keep track of the intemediate states that you have to concatenate 
    ---> the solution then is to scrap this structure with the create_block and define a custom layers that stores all the parameters in itself
    * at the same time you can re-define the convolution to make it periodic!
end 