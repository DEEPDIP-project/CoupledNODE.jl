using Lux: Lux
using LuxCore: AbstractLuxLayer
using FFTW: fft, ifft
using Tullio: @tullio

# required pieces:
# - downsampler
# - upsampler
# - activation
# - CCN (standard layer)
# notice that downsampler and upsampler have NO weights, so I can just code them as functions!

# expected input shape: [N, N, d, batch]
# expected output shape: [N, N, d, batch]

function create_lowpass_filter(T, cutoff, N)
    # TODO extend to multiple dimensions
    center = div(N, 2)

    # TODO I think I have to remove the boundaries

    # Compute the radial distance considering periodic boundaries
    radial_distances = zeros(T, N, N)
    for i in 1:N
        for j in 1:N
            di = min(abs(i - center), N - abs(i - center))
            dj = min(abs(j - center), N - abs(j - center))
            r = sqrt(di^2 + dj^2)
            radial_distances[i, j] = r
        end
    end

    # Create the 2D sinc kernel based on radial distances
    sinc_kernel = sinc.(2 * cutoff * radial_distances)
    # Normalize the kernel to conserve energy
    sinc_kernel ./= sum(sinc_kernel)

    # Do the fft of the kernel once
    K_f = fft(sinc_kernel)

    function sinc_interpolation(x)
        # Perform circular convolution using FFT (notice I am assuming PBC in both directions)
        X_f = fft(x)
        @tullio filtered_f[i, j, c, b] := X_f[i, j, c, b] .* K_f[i, j]
        #        filtered_f = X_f .* K_f       
        real(ifft(filtered_f))
    end
end

function create_CNOdownsampler(T::Type, D::Int, N::Int, down_factor::Int, cutoff)
    filtered_size = (((1:down_factor:N) for _ in 1:D)..., :, :)
    filter = create_lowpass_filter(T, cutoff, N)

    function CNOdownsampler(x)
        # Apply the lowpass filter
        x_filter = filter(x)
        # then take only the downsampled values
        @view x_filter[filtered_size...]
    end
end

function create_CNOupsampler(T::Type, D::Int, N::Int, up_factor::Int, cutoff)
    D_up = up_factor * (N - 1) + 1
    up_size = (D_up for _ in 1:D)
    filter = create_lowpass_filter(T, cutoff, D_up)

    function CNOdownsampler(x)
        # Enhance to the upsampled size
        x_up = zeros(T, up_size..., size(x)[end - 1], size(x)[end])
        x_up[1:up_factor:end, 1:up_factor:end, :, :] .= x
        # then apply the lowpass filter
        filter(x_up)
    end
end

function remove_BC(x)
    # TODO this is redundant with NN_padded_to_NN_nopad, but I want to use it like this
    @view x[2:(end - 1), 2:(end - 1), :, :]
end
