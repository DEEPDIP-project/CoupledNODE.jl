using Lux: Lux
using LuxCore: AbstractLuxLayer

# required pieces:
# - downsampler
# - upsampler
# - activation
# - CCN (standard layer)
# notice that downsampler and upsampler have NO weights, so I can just code them as functions!

# expected input shape: [N, N, d, batch]
# expected output shape: [N, N, d, batch]

function sinc_interpolation(x)
    x
end

function create_CNOdownsampler(D::Int,nin::Int, nout::Int)
    @assert nout < nin "In Downsampler nout must be smaller than nin" 
    @assert nin%nout == 0 "In Downsampler nin must be divisible by nout"
    one_every = div(nin, nout)
    filtered_size = (((1:one_every:nin) for _ in 1:D)..., :, :)

    function CNOdownsampler(x)
        # Apply the lowpass filter
        x_filter = sinc_interpolation(x)
        # then take only the downsampled values
        @view x_filter[filtered_size...]
    end
end

function create_CNOupsampler(D::Int,nin::Int, nout::Int)
    @assert nout > nin "In Upsampler nout must be larger than nin" 
    @assert nout%nin == 0 "In Upsampler nout must be divisible by nin"
    one_every = div(nin, nout)
    filtered_size = (((1:one_every:nin) for _ in 1:D)..., :, :)

    function CNOdownsampler(x)
        # Enhance to the upsampled size
        x_filter[filtered_size...]

        # then apply the lowpass filter
        sinc_interpolation(x_upsampled)
    end
end