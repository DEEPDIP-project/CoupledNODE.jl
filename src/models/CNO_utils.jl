using FFTW: fft, ifft

# This code defines utils for the Convolutional Neural Operators (CNO) architecture.
# It includes also the functionals for the downsampler, upsampler, and activation function.
# They will be fundamental for the CNOLayers defined in CNO.jl

function create_filter(T, grid, cutoff; sigma = 1, filter_type = "sinc")
    # TODO extend to multiple dimensions
    N = length(grid)
    N2 = Int(N / 2)
    _kernel = zeros(T, N, N)

    if filter_type == "gaussian"
        k = Int(cutoff)
        center = k / 2
        for i in 1:k
            for j in 1:k
                x = center - i
                y = center - j
                _kernel[i, j] = exp(-(x^2 + y^2) / (2 * sigma^2))
            end
        end
    elseif filter_type == "sinc"
        omega = cutoff * pi
        for x in (-N2 + 1):1:N2
            for y in (-N2 + 1):1:N2
                _kernel[x + N2, y + N2] = sinc(x * omega) * sinc(y * omega)
            end
        end
        _kernel = circshift(_kernel, (N2, N2))
    elseif filter_type == "lanczos"
        @warn "You can NOT use lanczos for CNO upsampling because this kernel has a low weight in the orthogonal directions, which is exactly the direction where we create high frequencies with a CNO."
        k = Int(cutoff)
        for i in 1:(2 * k + 1)
            for j in 1:(2 * k + 1)
                x = i - (k + 1)
                y = j - (k + 1)
                pos = sqrt(x^2 + y^2)
                _kernel[i, j] = sinc(pos) * sinc(pos / k)
            end
        end
    elseif filter_type == "identity"
        _kernel .= 1
    else
        error("Filter type not recognized")
    end

    # normalize the kernel
    _kernel = _kernel / sum(_kernel)

    # Do the fft of the kernel once
    K_f = fft(_kernel)

    function apply_fitler(x)
        # Perform circular convolution using FFT (notice I am assuming PBC in both directions)
        X_f = fft(x)
        filtered_f = X_f .* K_f
        real(ifft(filtered_f))
    end
end

function create_CNOdownsampler(
        T::Type, D::Int, N::Int, down_factor::Int, cutoff, filter_type = "sinc")
    grid = collect(0.0:(1.0 / (N - 1)):1.0)
    filtered_size = (((1:down_factor:N) for _ in 1:D)..., :, :)
    filter = create_filter(T, grid, cutoff, filter_type = filter_type)
    # The prefactor is the factor by which the energy is conserved (check 'Convolutional Neural Operators for robust and accurate learning of PDEs')
    prefactor = T(1 / down_factor^D)

    function CNOdownsampler(x)
        # Apply the lowpass filter
        x_filter = filter(x) * prefactor
        # then take only the downsampled values
        @view x_filter[filtered_size...]
    end
end

function expand_with_zeros(x, T, up_size, up_factor)
    x_up = zeros(T, up_size..., size(x)[end - 1], size(x)[end])
    x_up[1:up_factor:end, 1:up_factor:end, :, :] .= x
    return x_up
end

using ChainRules: ChainRulesCore, NoTangent
function ChainRulesCore.rrule(::typeof(expand_with_zeros), x, T, up_size, up_factor)
    y = expand_with_zeros(x, T, up_size, up_factor)
    function expand_with_zeros_pb(ȳ)
        x̄ = zeros(T, size(x))
        x̄ .= ȳ[1:up_factor:end, 1:up_factor:end, :, :]
        return NoTangent(), x̄, NoTangent(), NoTangent(), NoTangent()
    end
    return y, expand_with_zeros_pb
end

function create_CNOupsampler(
        T::Type, D::Int, N::Int, up_factor::Int, cutoff, filter_type = "sinc")
    D_up = up_factor * N
    up_size = (D_up for _ in 1:D)
    grid_up = collect(0.0:(1.0 / (D_up - 1)):1.0)
    filter = create_filter(T, grid_up, cutoff, filter_type = filter_type)

    function CNOupsampler(x)
        # Enhance to the upsampled size
        x_up = expand_with_zeros(x, T, up_size, up_factor)
        # then apply the lowpass filter
        filter(x_up)
    end
end

function remove_BC(x)
    # TODO this is redundant with NN_padded_to_NN_nopad, but I want to use it like this
    @view x[2:(end - 1), 2:(end - 1), :, :]
end

function create_CNOactivation(T::Type, D::Int, N::Int, cutoff;
        activation_function = identity, filter_type = "sinc")
    # the activation function is applied like this:
    # upsamplex2 -> apply activation -> downsamplex2 
    us = create_CNOupsampler(T, D, N, 2, cutoff, filter_type)
    ds = create_CNOdownsampler(T, D, N * 2, 2, cutoff, filter_type)
    function CNOactivation(x)
        ds(activation_function(us(x)))
    end
end

function ch_to_ranges(arr::Vector{Int})
    # TODO write the docstring for this
    # it returns a vector containing range_k
    ranges = Vector{UnitRange{Int}}()
    start = 1
    for n in arr
        push!(ranges, start:(start + n - 1))
        start += n
    end
    return ranges
end

function ch_to_bottleneck_ranges(bot_d::Vector{Int}, ch_size::Vector{Int})
    # TODO write the docstring for this
    # it returns a vector containing tuple of (range_k, bottleneck_extra_info)
    @assert length(bot_d)==length(ch_size) "The bottleneck depth and the channel size must have the same length"
    ranges = Vector{Any}()
    start = 1
    for (i, nblock) in enumerate(bot_d)
        this_bottleneck = ()
        # these are all the resblocks
        for x in 1:(2 * (nblock - 1))
            this_bottleneck = (this_bottleneck..., (start:(start + ch_size[i] - 1),))
            start = start + ch_size[i]
        end
        # then this is the last block
        this_bottleneck = (this_bottleneck..., (start:(start + ch_size[i] - 1),))
        push!(ranges, this_bottleneck)
        start += ch_size[i]
    end
    return ranges
end
