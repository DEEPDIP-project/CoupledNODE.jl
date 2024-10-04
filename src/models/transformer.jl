using Lux: Lux



struct AttentionLayer{F} <: Lux.AbstractExplicitLayer
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
        init_weight = Lux.glorot_uniform)
    @assert N % patch_size == 0 "N must be divisible by patch_size"
    n_patches = (div(N, patch_size))^d
    dh = div(emb_size, n_heads)
    println("In the constructor")
    AttentionLayer(N, d, emb_size, patch_size, n_patches, n_heads, dh, init_weight)
end

# We also need to specify how to initialize the parameters and states. 

function Lux.initialparameters(rng::AbstractRNG,
        (; N, d, emb_size, patch_size, dh, n_heads, init_weight)::AttentionLayer)
    (;
        # the attention weights have this size
        wQ = init_weight(rng, n_heads, emb_size+1, dh),
        wK = init_weight(rng, n_heads, emb_size+1, dh),
        wV = init_weight(rng, n_heads, emb_size+1, dh),
        # then the embedding operator
        Ew = init_weight(rng, patch_size*patch_size*d, emb_size),
        Eb = zeros(Float32, emb_size),
        # then the multihead attention
        wU = init_weight(rng, emb_size, emb_size),
    )
end

function Lux.initialstates(rng::AbstractRNG, (; N, d, emb_size, patch_size, n_patches, dh, n_heads)::AttentionLayer)
    (;
        N = N,
        d = d,
        emb_size = emb_size,
        patch_size = patch_size,
        n_patches = n_patches,
        n_heads = n_heads,
        dh = dh,
        sqrtDh = sqrt(dh),
    )
end
function Lux.parameterlength((; N, d)::AttentionLayer)
    3 * N * N * d
end
Lux.statelength(::AttentionLayer) = 8

# We now define how to pass inputs through an attention layer assuming the
# following:
# - Input size: `(Nxyz , nchannels, nsample)`, where we allow multichannel input to for example allow the user to pass (u, u^2) or even to pass (u,v)
# - Output size: `(Nxyz , nsample)` where we assumed monochannel output, so we dropped the channel dimension

# This is what each layer does:
function ((;)::AttentionLayer)(x, params, state)
    d = state.d
    np = state.n_patches
    ps = state.patch_size
    dh = state.dh
    sqrtDh = state.sqrtDh
    n_heads = state.n_heads

    Ew = params.Ew
    Eb = params.Eb
    wQ = params.wQ
    wK = params.wK
    wV = params.wV

    # (1) Split the image into patches 
    x_patches = [x[ps*i+1:ps*(i+1), ps*j+1:ps*(j+1), :, :] for i in 0:div(size(x, 1)-1, ps), j in 0:div(size(x, 2)-1, ps)]
    println("Split the image into patches of size $ps. Those are the sizes")
    println(size(x))
    println(size(x_patches))
    println(size(first(x_patches)))

    # (2) flatten the patches
    # TODO: do NOT use reshape here
    x_pflat = [reshape(p, ps*ps*d, : ) for p in x_patches] 
    #x_pflat = [vec(p) for p in x_patches] 
    println("flatten the patches. Those are the sizes")
    println(size(x_pflat))
    println(size(first(x_pflat)))

    # (3) project the patches onto the embedding space
    x_emb = [Ew' * p .+ Eb for p in x_pflat]
    println("project the patches onto the embedding space. Those are the sizes")
    println(size(x_emb))
    println(size(first(x_emb)))

    # (4) positional embedding
    # notice that we use 1D positional embedding, as suggested [here](https://arxiv.org/pdf/2010.11929)
    x_emb = [cat(p,zeros(1,size(p)[2:end]...)*i; dims=1) for (i,p) in enumerate(x_emb)]
    println("positional embedding. Those are the sizes")
    println(size(x_emb))
    println(size(first(x_emb)))

    # (5) compute the attention scores
    They are not computed correctly since they have the wrong dimension (it should be dh instead of embed_dim)
    Q = [x_emb * wQ[i] for i in 1:n_heads]
    K = [x_emb * wK[i] for i in 1:n_heads]
    V = [x_emb * wV[i] for i in 1:n_heads]
    println("compute the attention scores. Those are the sizes")
    println(size(Q))
    println(size(first(Q)))
    println(size(first(first((Q)))))
    # Then I unroll them into objects of shape [nheads, n_patches, dh, batch]
    println("-------")
    Q1 = Array{Float32,4}(undef,n_heads,np,dh,size(x)[end])
    for i in 1:n_heads
        for patchi in 1:np, q in Q[i]
            println(size(q))
            println(size(Q1))
            Q1[i,patchi,:,:] .= q
        end
    end
    println("-------")
    println(size(Q1))

    A = [[Lux.softmax(q * permutedims(k, (2, 1)) / sqrtDh) for q in Q[i], k in K[i]] for i in 1:n_heads]
   # for i in 1:n_heads
   #     for q in Q[i], k in K[i]
   #         println("-------")
   #         println(size(q))
   #         println(size(k))
   #         println(size(permutedims(k, (2, 1)) * q / sqrtDh))
   #     end
   # end
    println("A has this shape")
    println(size(A))
    println(size(first(A)))
    println(size(first(first(A))))
    SA = A .* V
    println("SA has this shape")
    println(size(SA))
    println(size(first(A)))
    println(size(first(first(A))))

    MSA = cat(SA; dims=3)
    println(size(A))
    println(size(SA))
    println(size(MSA))

    MSA=MSA* UMSA

    # Attention layer does not modify state
    MSA, state
end
