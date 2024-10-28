using Lux: Lux, relu, leakyrelu
using NNlib: tanh_fast
using LuxCore: AbstractLuxLayer
using Tullio

# Observation: the original CNO paper basically assumes PBC everywhere.

# expected input shape: [N, N, d, batch]
# expected output shape: [N, N, d, batch]

struct CNO{F} <: AbstractLuxLayer
    T::Type
    N::Int
    D::Int
    cutoff::Union{Float32,Float64} #TODO make it same type as T
    activations::Array
    down_factors::Array
    down_ch::Array
    up_factors::Array
    up_ch::Array
    k_radii::Array
    use_biases::Array
    use_batchnorms::Array
    bottleneck_depths::Array
    bottlenecks_radii::Array
    bottlenecks_ch::Array
    filter_type::String
    init_weight::F
end

function create_CNO(
        ;
        N,
        D,
        cutoff,
        ch_sizes,
        down_factors,
        k_radii,
        activations = nothing,
        use_biases = nothing,
        use_batchnorms = nothing,
        bottleneck_depths = nothing,
        bottlenecks_radii = nothing,
        filter_type="sinc",
        T = Float32,
        init_weight = Lux.kaiming_uniform)
    @assert length(ch_sizes) == length(down_factors) == length(k_radii) "The size of the input arrays must be consistent"
    if activations == nothing
        activations = fill(leakyrelu, length(ch_sizes))
    end
    if use_biases == nothing
        use_biases = fill(false, length(ch_sizes))
    else
        @warn "Bias is not implemented"
    end
    if use_batchnorms == nothing
        use_batchnorms = fill(false, length(ch_sizes))
    else
        @warn "Batchnorm is not implemented yet"
    end
    if bottleneck_depths == nothing
        bottleneck_depths = fill(2, length(ch_sizes)+1)
    else
        @assert all(bottleneck_depths .>= 2) "The bottleneck depth must be at least 2"
        @assert length(bottleneck_depths) == length(ch_sizes)+1 "The number of bottleneck depths must be equal to the number of up/dw layers + 1"
    end
    if bottlenecks_radii == nothing
        bottlenecks_radii = fill(3, length(ch_sizes)+1)
    else
        @assert all(bottlenecks_radii .>= 0) "The bottleneck radii must be at least 0"
        @assert length(bottlenecks_radii) == length(ch_sizes)+1 "The number of bottleneck radii must be equal to the number of up/dw layers + 1"
    end
     
    up_factors = reverse(down_factors)
    bottlenecks_ch = [D; ch_sizes]
    down_ch = ch_sizes
    up_ch = reverse([D; ch_sizes[1:end-1]])

    CNO(T, N, D, cutoff, activations, down_factors, down_ch, up_factors, up_ch, k_radii, use_biases, use_batchnorms, bottleneck_depths, bottlenecks_radii, bottlenecks_ch, filter_type, init_weight)
end

function Lux.initialparameters(rng::AbstractRNG,
        (; T, N, D, down_factors, down_ch, up_factors, up_ch, k_radii, bottleneck_depths, bottlenecks_radii, bottlenecks_ch, filter_type, init_weight)::CNO)

    # Count the number of bottlenecks kernels
    # There are two types of bottleneck blocks: residual and standard. Here we always use the residual, except for the last one (TODO make this controllable)
    # so we have for i=1:length(bottleneck_depths):
    #   - bottleneck_depths[i]-1 residual blocks containing each 2*ch_size kernels
    #   - 1 standard block containing ch_size kernels
    nb = 0
    for i in eachindex(bottleneck_depths)
        nb += 2*(bottleneck_depths[i]-1)*bottlenecks_ch[i]
        nb += bottlenecks_ch[i]
    end
    (;
        # Notice that those kernels are in the k-space, because we don't want to transform them every time
        # [!] NO ^  you can not keep them in Fourier space otherwise they are non-local and impossible to mask correctly!
        # Also notice that the kernels contain a zero padding in order to have the same size and allow for serialization (otherwise Zygote complains)
        # they will be adapted to the correct size by using masks
        # notice also that there is a kernel for every output channel 
        # TODO kernel biases are missing
        down_k = init_weight(rng, T, sum(down_ch), N,N),
        up_k = init_weight(rng, T, sum(up_ch), N,N),
        bottlenecks = init_weight(rng, T, nb, N,N),
    )
end

function Lux.initialstates(rng::AbstractRNG,
        (; T, N, D, cutoff, activations, down_factors, down_ch, up_factors, up_ch, k_radii, bottleneck_depths, bottlenecks_radii, bottlenecks_ch, filter_type)::CNO)
    downsamplers = []
    activations_down = []
    Nd = N
    # Define downsampling, upsampling and activation functions
    for (i,df) in enumerate(down_factors)
        push!(activations_down, create_CNOactivation(T, D, Nd, cutoff, activation_function = activations[i], filter_type=filter_type))
        push!(downsamplers, create_CNOdownsampler(T, D, Nd, df, cutoff, filter_type))
        Nd = Int(Nd/df)
    end
    upsamplers = []
    activations_up = []
    for (i,uf) in enumerate(up_factors)
        push!(activations_up, create_CNOactivation(T, D, Nd, cutoff, activation_function = reverse(activations)[i], filter_type=filter_type))
        push!(upsamplers, create_CNOupsampler(T, D, Nd, uf, cutoff, filter_type))
        Nd = Int(Nd*uf)
    end
    # compute the masks to adapt the upsampling/downsampling kernels to the correct size
    masks_down = []
    for (i,df) in enumerate(down_factors)
        mask = zeros(T, N, N)
        k = 2*k_radii[i]+1
        mask[1:k, 1:k] .= 1
        push!(masks_down, mask)
    end
    masks_up = []
    for (i,uf) in enumerate(up_factors)
        mask = zeros(T, N, N)
        k = 2*reverse(k_radii)[i]+1
        mask[1:k, 1:k] .= 1
        push!(masks_up, mask)
    end
    # compute the masks for the bottleneck kernels
    # Notice that all the convolutions in the same bottleneck will have the same size (TODO make this controllable by the user)
    masks_bottlenecks = []
    for (i,bs) in enumerate(bottleneck_depths)
        mask = zeros(T, N, N)
        k = 2*bottlenecks_radii[i]+1
        mask[1:k, 1:k] .= 1
        push!(masks_bottlenecks, mask)
    end
    (;
        T = T,
        N = N,
        D = D,
        cutoff=cutoff,
        up_ch_ranges = ch_to_ranges(up_ch),
        down_ch_ranges = ch_to_ranges(down_ch),
        # The first bottleneck layer has ch_sizes of the input
        bottleneck_ranges = ch_to_bottleneck_ranges(bottleneck_depths, bottlenecks_ch),
        down_factors = down_factors,
        up_factors = up_factors,
        downsamplers = downsamplers,
        upsamplers = upsamplers,
        activations_up = activations_up,
        activations_down = activations_down,
        masks_down = masks_down,
        masks_up = masks_up,
        masks_bottlenecks = masks_bottlenecks,
    )
end

function Lux.parameterlength((;
        N, D, k_radii, bottlenecks_radii, bottleneck_depths)::CNO)
    param_length = 0
    for k in k_radii
        param_length += (2*k+1)^2
    end
    for i in eachindex(bottleneck_depths)
        param_length += bottleneck_depths[i] * (2*bottlenecks_radii[i]+1)^2
    end
    # TODO bias should be counted as well 
    param_length
end

function Lux.statelength((;down_factors)::CNO) 
    # TODO count and update this number
    4 + 2*length(down_factors)
end

function ((;)::CNO)(x, params, state)
    T = state.T
    N = state.N
    D = state.D
    k_down = params.down_k
    k_up = params.up_k
    k_bottlenecks = params.bottlenecks
    downsampling_layers = state.downsamplers
    upsampling_layers = state.upsamplers
    activations_layers_up = state.activations_up
    activations_layers_down = state.activations_down
    masks_down = state.masks_down
    masks_up = state.masks_up
    masks_bottlenecks = state.masks_bottlenecks
    up_ch_ranges = state.up_ch_ranges
    down_ch_ranges = state.down_ch_ranges
    bottleneck_ranges = state.bottleneck_ranges

    # we have to keep track of each downsampled state
    intermediate_states = []

    y = copy(x)
    intermediate_states = vcat(intermediate_states, [y] )
    # * Downsampling blocks
    for (i,(ds,da)) in enumerate(zip(downsampling_layers, activations_layers_down))
        # 1) convolve
        y = apply_masked_convolution(y, k = k_down[down_ch_ranges[i],:,:], mask=masks_down[i])
        # 2) activate
        y = da(y)
        # 3) downsample
        y = ds(y)
        # 4) store intermediates
        intermediate_states = vcat(intermediate_states, [y])
    end

    # * Bottleneck blocks
    # they are n-1 residual blocks each consisting of conv->act->conv
    # followed by 1 non-residual blocks of conv->act
    # The last block takes as input the cat with the downsampler 
    # and it passes its output to the upsample (except the last one).
    # Insted, the last bottleneck is applied directly to the compressed data
    # -> apply the n-1 residual blocks
    mask = masks_bottlenecks[end]
    k_residual= bottleneck_ranges[end][1:end-1]
    y = apply_residual_blocks(y, k_bottlenecks, k_residual, mask, activations_layers_up[1])

    # -> last (non-residual) block of the bottleneck
    y = apply_masked_convolution(y, k = k_bottlenecks[bottleneck_ranges[end][end]...,:,:], mask=masks_bottlenecks[end])
    y = activations_layers_up[1](y)

    ## Now I have to apply the bottlenecks to all the intermediates (except the last one which is already processed as y)
    bottlenecks_out = []
    for (i,istate) in enumerate(intermediate_states[1:end-1])
        mask = masks_bottlenecks[i]
        k_residual= bottleneck_ranges[i][1:end-1]
        b = apply_residual_blocks(istate, k_bottlenecks, k_residual, mask, activations_layers_down[i])
        bottlenecks_out = vcat(bottlenecks_out, [b])
    end
    # Notice that they are stored [top...bottom], so I have to reverse them 
    bottlenecks_out = reverse(bottlenecks_out)

    # * Upsampling blocks
    for (i,(us,ua)) in enumerate(zip(upsampling_layers, activations_layers_up))
        # 1) convolve
        y = apply_masked_convolution(y, k = k_up[up_ch_ranges[i],:,:], mask=masks_up[i])
        # 2) activate
        y = ua(y)
        # 3) upsample
        y = us(y)
        # 4) concatenate with the corresponding bottleneck
        y = cat(y, bottlenecks_out[i], dims=D+1)
        # 5) apply the last bottleneck
        # ! do not forget to reverse the bottleneck ranges
        y = apply_masked_convolution(y, k = k_bottlenecks[reverse(bottleneck_ranges)[i+1][end]...,:,:], mask=masks_bottlenecks[i])
        y = reverse(activations_layers_down)[i](y)
    end

    

    y, state
end

using NNlib: pad_zeros
function convolve(x,k)
    # following [this](https://datascience.stackexchange.com/questions/68045/how-are-the-channels-handled-in-cnn-is-it-independently-processed-or-fused)
    # I basically have to convolve ch_out kernels to each ch_in and sum them (?TODO? sure about sum?) 
    # ! the index can not have 'long' names like ch_in
    @tullio ffty[i,j,c,b] := fft(x)[i,j,a,b] * fft(k)[c,i,j]
    real(ifft(ffty))
end



function apply_residual_blocks(y, k_bottlenecks, k_residual, mask, activation)
    ## Loop in steps of 2 because I have to consider pairs of kernels for a residual block
    for ik in 1:2:length(k_residual)
        # Store the input to the residual block
        y0 = copy(y)
        
        # Get the kernels
        ka = k_bottlenecks[k_residual[ik]..., :, :]
        kb = k_bottlenecks[k_residual[ik+1]..., :, :]
        
        # Apply masks to the kernels
        k2a = mask_kernel(ka, mask)
        k2b = mask_kernel(kb, mask)
        
        # Adjust the kernel sizes to match the input dimensions
        k3a = k2a[:, 1:size(y0)[1], 1:size(y0)[2]]
        k3b = k2b[:, 1:size(y0)[1], 1:size(y0)[2]]
        
        # Apply the first convolution
        y = convolve(y, k3a)
        
        # Activate
        y = activation(y)
        
        # Apply the second convolution
        y = convolve(y, k3b)
        
        # Residual sum
        y = y .+ y0
    end
    
    return y
end


function apply_masked_convolution(y; k, mask )
    # to get the correct k i have to reshape+mask+trim
    # TODO: i don't like this...
    # ! Zygote does not like that you reuse variable names so, this makes it even uglier with the definition of k2 and k3
    # ! also Zygote wants the mask to be explicitely defined as a vector so i have to pull it out from the tuple via mask=masks[i]

    # Apply the mask to the kernel
    k2 = mask_kernel(k, mask)
    
    # Adjust the kernel size to match the input dimensions
    k3 = k2[:, 1:size(y)[1], 1:size(y)[2]]
    
    # Apply the convolution
    y = convolve(y, k3)
    
    return y
end

function mask_kernel(k, mask)
    @tullio k2[c, x, y] := k[c, x, y] * mask[x, y]
end