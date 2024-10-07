using Lux: Lux

struct AttentionLayer{F} <: Lux.AbstractExplicitLayer
    T::Type
    N::Int
    d::Int
    emb_size::Int
    patch_size::Int
    n_patches::Int
    n_heads::Int
    dh::Int
    init_weight::F
end

function AttentionLayer(
        N::Int,
        d::Int,
        emb_size::Int,
        patch_size::Int,
        n_heads::Int;
        T = Float32,
        init_weight = Lux.glorot_uniform)
    @assert N % patch_size==0 "N must be divisible by patch_size"
    n_patches = (div(N, patch_size))^d
    dh = div(emb_size, n_heads)
    println("In the constructor")
    AttentionLayer(T, N, d, emb_size, patch_size, n_patches, n_heads, dh, init_weight)
end

# We also need to specify how to initialize the parameters and states. 

function Lux.initialparameters(rng::AbstractRNG,
        (; T, N, d, emb_size, patch_size, n_patches,
            dh, n_heads, init_weight)::AttentionLayer)
    (;
        # the attention weights have this size
        wQ = init_weight(rng, T, n_heads, dh, emb_size + 1),
        wK = init_weight(rng, T, n_heads, dh, emb_size + 1),
        wV = init_weight(rng, T, n_heads, dh, emb_size + 1),
        # then the embedding operator
        Ew = init_weight(rng, T, emb_size, patch_size * patch_size * d),
        Eb = zeros(T, emb_size),
        # then the multihead attention
        U = init_weight(rng, T, N * N * d, n_patches * n_heads * dh)
    )
end

function Lux.initialstates(rng::AbstractRNG,
        (; T, N, d, emb_size, patch_size, n_patches, dh, n_heads)::AttentionLayer)
    (;
        T = T,
        N = N,
        d = d,
        emb_size = emb_size,
        patch_size = patch_size,
        n_patches = n_patches,
        n_heads = n_heads,
        dh = dh,
        sqrtDh = T(sqrt(dh))
    )
end
function Lux.parameterlength((;
        N, d, n_heads, dh, emb_size, patch_size, n_patches)::AttentionLayer)
    3 * n_heads * dh * (emb_size + 1) + patch_size * patch_size * d * emb_size + emb_size +
    N * N * d * n_patches * n_heads * dh
end
Lux.statelength(::AttentionLayer) = 9

# This is what each layer does:
# expected input shape: [N, N, d, batch]
# expected output shape: [N, N, d, batch]
function ((;)::AttentionLayer)(x, params, state)
    N = state.N
    d = state.d
    np = state.n_patches
    ps = state.patch_size
    dh = state.dh
    sqrtDh = state.sqrtDh
    n_heads = state.n_heads
    emb_size = state.emb_size

    Ew = params.Ew
    Eb = params.Eb
    wQ = params.wQ
    wK = params.wK
    wV = params.wV
    U = params.U

    # (1) Split the image into patches 
    #x_patches = [x[ps*i+1:ps*(i+1), ps*j+1:ps*(j+1), :, :] for i in 0:div(size(x, 1)-1, ps), j in 0:div(size(x, 2)-1, ps)]
    num_patches_h = div(N, ps)
    num_patches_w = div(N, ps)
    x_patches = [x[(i * ps + 1):(i * ps + ps), (j * ps + 1):(j * ps + ps), :, :]
                 for i in 0:(num_patches_h - 1), j in 0:(num_patches_w - 1)]

    # (2) flatten the patches
    # TODO: do NOT use reshape here
    x_pflat = [reshape(p, ps * ps * d, :) for p in x_patches]

    #x_pflat = [ones(state.T, ps*ps*d, size(x)[end]) for p in x_patches] 

    # (3) project the patches onto the embedding space
    x_emb = [Ew * p .+ Eb for p in x_pflat]

    # (4) positional embedding
    # notice that we use 1D positional embedding, as suggested [here](https://arxiv.org/pdf/2010.11929)
    x_lemb = [cat(p, ones(state.T, 1, size(p)[2:end]...) * i; dims = 1)
              for (i, p) in enumerate(x_emb)]
    #x_emb = [ones(state.T, emb_size+1, size(x)[end]) for p in x_emb] 

    # (5) compute the attention scores
    # [!] notice that you can not reuse some variable names otherwise Zygote gets confused
    Q0 = [wQ[i, :, :] * x_lemb[patchi] for i in 1:n_heads, patchi in 1:np]
    K0 = [wK[i, :, :] * x_lemb[patchi] for i in 1:n_heads, patchi in 1:np]
    V0 = [wV[i, :, :] * x_lemb[patchi] for i in 1:n_heads, patchi in 1:np]
    # Reshape Q, K, V to match desired output dimensions
    Q = reshape(vcat(Q0...), (n_heads, np, dh, size(x)[end]))
    K = reshape(vcat(K0...), (n_heads, np, dh, size(x)[end]))
    V = reshape(vcat(V0...), (n_heads, np, dh, size(x)[end]))

    # (6) Compute attention scores without mutations
    A = [Lux.softmax(Q[i, p, :, :] .* K[i, p, :, :] / sqrtDh) for i in 1:n_heads, p in 1:np]
    A = reshape(vcat(A...), (n_heads, np, dh, size(x)[end]))
    SA = A .* V

    # (7) multihead attention
    MSA = reshape(SA, n_heads * np * dh, size(x)[end])
    MSA = U * MSA
    MSA = reshape(MSA, size(x)...)

    # Attention layer does not modify state
    MSA, state
end
