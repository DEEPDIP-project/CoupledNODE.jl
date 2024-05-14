# # Fourier neural operator architecture
#
# Now let's implement the Fourier Neural Operator (FNO).
# A Fourier layer $u \mapsto w$ is given by the following expression:
#
# $$
# w(x) = \sigma \left( z(x) + W u(x) \right)
# $$
#
# where $\hat{z}(k) = R(k) \hat{u}(k)$ for some weight matrix collection
# $R(k) \in \mathbb{C}^{n_\text{out} \times n_\text{in}}$. The important
# part is the following choice: $R(k) = 0$ for $\| k \| > k_\text{max}$
# for some $k_\text{max}$. This truncation makes the FNO applicable to
# any discretization as long as $K > k_\text{max}$, and the same parameters
# may be reused.
#
# The deep learning framework [Lux](https://lux.csail.mit.edu/) let's us define
# our own layer types. Everything should be explicit ("functional
# programming"), including random number generation and state modification. The
# weights are stored in a vector outside the layer, while the layer itself
# contains information for construction the network.

import Lux
import Lux: Dense, gelu
if CUDA.functional()
    import LuxCUDA
end
import FFTW: fft, ifft, irfft
import Random: AbstractRNG

struct FourierLayer{A, F} <: Lux.AbstractExplicitLayer
    dim_to_fft::Tuple{Vararg{Int}}
    Nxyz::Tuple{Vararg{Int}}
    kmax::Int
    cin::Int
    cout::Int
    σ::A
    init_weight::F
end

function FourierLayer(
        dim_to_fft::Tuple{Vararg{Int}},
        Nxyz::Tuple{Vararg{Int}},
        kmax,
        ch::Pair{Int, Int};
        σ = identity,
        init_weight = Lux.glorot_uniform)
    FourierLayer(dim_to_fft, Nxyz, kmax, first(ch), last(ch), σ, init_weight)
end

# We also need to specify how to initialize the parameters and states. 

function Lux.initialparameters(rng::AbstractRNG,
        (; Nxyz, kmax, cin, cout, init_weight)::FourierLayer)
    mydims = length(Nxyz)
    kgrid = ntuple(d -> kmax + 1, mydims)
    (;
        spatial_weight = init_weight(rng, cout, cin),
        # the last dimension of size 2 is for real and imaginary part
        spectral_weights = init_weight(rng, kgrid..., cout, cin, 2)
    )
end
function Lux.initialstates(rng::AbstractRNG, (; Nxyz, kmax, cin)::FourierLayer)
    mydims = length(Nxyz)
    kgrid = ntuple(d -> kmax + 1, mydims)
    (;
        mydims = mydims,
        kgrid = kgrid,
        placeholder = ntuple(d -> 1, mydims),
        ikeep = ntuple(d -> 1:(kmax + 1), mydims),
        nkeep = ntuple(d -> kmax + 1, mydims),
        last_dim = mydims + 3,
        # pad with a 0 on the left and a N-k-1 on the right for each dimension
        pad = ntuple(d -> d % 2 == 0 ? Nxyz[1] - kmax - 1 : 0, mydims * 2)
    )
end
function Lux.parameterlength((; Nxyz, kmax, cin, cout)::FourierLayer)
    cout * cin + (kmax + 1)^length(Nxyz) * 2 * cout * cin
end
Lux.statelength(::FourierLayer) = 7 #TODO: make it a function

# We now define how to pass inputs through Fourier layer, assuming the
# following:
#
# - Input size: `(Nxyz , nchannels, nsample)`, where we allow multichannel input to for example allow the user to pass (u, u^2) or even to pass (u,v)
# - Output size: `(Nxyz , nsample)` where we assumed monochannel output, so we dropped the channel dimension

# This is what each Fourier layer does:
function ((; dim_to_fft, Nxyz, kmax, cout, cin, σ)::FourierLayer)(x, params, state)
    ## Destructure params
    W = params.spatial_weight
    W = reshape(W, state.placeholder..., cout, cin)
    R = params.spectral_weights
    ## The real and imaginary parts of R are stored in two separate channels
    R = selectdim(R, state.last_dim, 1) .+ im .* selectdim(R, state.last_dim, 2)

    println("x: ", size(x), "\n")
    ## Spatial part (applied point-wise)
    y = reshape(x, Nxyz..., 1, cin, :)
    # sum over the channel dimension
    y = sum(W .* y; dims = state.last_dim - 1)
    y = reshape(y, Nxyz..., cout, :)

    # Spectral part (applied mode-wise)
    #
    # Steps:
    # - go to complex-valued spectral space
    z = fft(x, dim_to_fft)
    # - chop off high wavenumbers
    z = z[state.ikeep..., :, :]
    # - multiply with weights mode-wise
    z = reshape(z, state.nkeep..., 1, cin, :)
    z = sum(R .* z; dims = state.last_dim - 1)
    z = reshape(z, state.nkeep..., cout, :)
    # - pad with zeros to restore original shape
    z = Lux.pad_zeros(z, state.pad; dims = dim_to_fft)
    # - go back to real valued spatial representation
    z = real.(fft(z, dim_to_fft))

    # Outer layer: Activation over combined spatial and spectral parts
    # Note: Even though high wavenumbers are chopped off in `z` and may
    # possibly not be present in the input at all, `σ` creates new high
    # wavenumbers. High wavenumber functions may thus be represented using a
    # sequence of Fourier layers. In this case, the `y`s are the only place
    # where information contained in high input wavenumbers survive in a
    # Fourier layer.
    v = σ.(y .+ z)

    # Fourier layer does not modify state
    v, state
end

# Function to create the model
function create_fno_model(kmax_fno, ch_fno, σ_fno, grid, input_channels = (u -> u,))
    # from the grids I can get the dimension
    dim = grid.dim
    ch_dim = dim + 1
    if dim == 1
        dim_to_fft = (1,)
        Nxyz = (grid.nx,)
        # permutations to toss around the channel dimesion
        ch_first = (2, 1, 3)
        ch_back = (2, 1, 3)
    elseif dim == 2
        dim_to_fft = (1, 2)
        Nxyz = (grid.nx, grid.ny)
        ch_first = (3, 1, 2, 4)
        ch_back = (2, 3, 1, 4)
    elseif dim == 3
        dim_to_fft = (1, 2, 3)
        Nxyz = (grid.nx, grid.ny, grid.nz)
        ch_first = (4, 1, 2, 3, 5)
        ch_back = (2, 3, 4, 1, 5)
    else
        error("Only 1D, 2D, and 3D grids are supported.")
    end

    # Add number of input channels
    # [!] you can use this first layer to pass (u, u^2), (u,v) or any custom Chain
    ch_fno = [length(input_channels); ch_fno]

    return Chain(
        input_channels[1],
        x -> let y = x
            println(" y:", size(y), "\n")
        end,
        (FourierLayer(
             dim_to_fft, Nxyz, kmax_fno[i], ch_fno[i] => ch_fno[i + 1]; σ = σ_fno[i]) for
        i in eachindex(σ_fno))...,
        # Put the channel dimension in the first position to apply a dense layer
        u -> permutedims(u, ch_first),
        Dense(ch_fno[end] => 2 * ch_fno[end], gelu),
        # in the end I will have a single channel
        Dense(2 * ch_fno[end] => 1; use_bias = false),
        u -> permutedims(u, ch_back),
        # drop the channel dimension
        u -> dropdims(u, dims = ch_dim)
    )
end
