using Lux: Lux
using LuxCore: AbstractLuxLayer
using FFTW: fft, ifft, fftshift, ifftshift
using ImageFiltering: imfilter, Kernel

# required pieces:
# - downsampler
# - upsampler
# - activation
# - CCN (standard layer)
# notice that downsampler and upsampler have NO weights, so I can just code them as functions!

# expected input shape: [N, N, d, batch]
# expected output shape: [N, N, d, batch]

function create_lowpass_filter(T, k, grid, sigma=1)
    # TODO there is something wrong in the sinc filter. I replace it with a gaussian for now

    # TODO extend to multiple dimensions
    N = length(grid)
    _kernel = zeros(T, N, N)

    center = k/2
    for i in 1:k
        for j in 1:k
            x = center - i
            y = center - j
            _kernel[i, j] = exp(-(x^2 + y^2) / (2 * sigma^2))
        end
    end
    # normalize the kernel
    _kernel = _kernel / sum(_kernel)

    # Do the fft of the kernel once
    K_f = fft(_kernel)

    function sinc_interpolation(x)
        # Perform circular convolution using FFT (notice I am assuming PBC in both directions)
        X_f = fft(x)
        filtered_f = X_f .* K_f       
        real(ifft(filtered_f))
        x
    end
    
end

function create_sinc_filter(T, cutoff, grid)
    # TODO extend to multiple dimensions
    N = length(grid)
    sinc_kernel = zeros(T, N, N)

    # Create the sinc kernel
    for i in 1:N
        for j in 1:N
            sinc_kernel[i, j] = sinc(2 * cutoff* grid[i]) * sinc(2 * cutoff* grid[j])
        end
    end
    # normalize the kernel
    sinc_kernel = sinc_kernel / sum(sinc_kernel)

    # Do the fft of the kernel once
    K_f = fft(sinc_kernel)

    function sinc_interpolation(x)
        # Perform circular convolution using FFT (notice I am assuming PBC in both directions)
        X_f = fft(x)
        filtered_f = X_f .* K_f       
        real(ifft(filtered_f))
    end
end



function create_CNOdownsampler(T::Type, D::Int, down_factor::Int, cutoff, grid)
    N = length(grid)
    filtered_size = (((1:down_factor:N) for _ in 1:D)..., :, :)
    filter = create_lowpass_filter(T, cutoff, grid)
    # The prefactor is the factor by which the energy is conserved (check 'Convolutional Neural Operators for robust and accurate learning of PDEs')
    prefactor = 1 / down_factor^D

    function CNOdownsampler(x)
        # Apply the lowpass filter
        x_filter = filter(x)*prefactor
        # then take only the downsampled values
        @view x_filter[filtered_size...]
    end
end

function create_CNOupsampler(T::Type, D::Int, up_factor::Int, cutoff, grid)
    N = length(grid)
    D_up = up_factor * N
    up_size = (D_up for _ in 1:D)
    grid_up = collect(0.0:1.0/(D_up - 1):1.0)
    filter = create_lowpass_filter(T, cutoff, grid_up)

    function CNOupsampler(x)
        # Enhance to the upsampled size
        x_up = zeros(T, up_size..., size(x)[end - 1], size(x)[end])
        x_up[1:up_factor:end, 1:up_factor:end, :, :] .= x #TODO do this without mutations
        # then apply the lowpass filter
        filter(x_up)
        #x_up.-filter(x_up)
    end
end

function remove_BC(x)
    # TODO this is redundant with NN_padded_to_NN_nopad, but I want to use it like this
    @view x[2:(end - 1), 2:(end - 1), :, :]
end
