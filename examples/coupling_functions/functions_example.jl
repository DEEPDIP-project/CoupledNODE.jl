f_o(u) = @. u.*(0.0.-0.8.*log.(u))
function observation()
    f_ND = create_NODE_obs()
    trange = (0.0f0, 6.0f0)
    p0 = [0.01]
    # define the observation from a NeuralODE
    obs_node = NeuralODE(f_ND, trange, Tsit5(), adaptive=false, dt=0.01, saveat=0.01)
    th_e, st_e = Lux.setup(rng, f_ND)
    return Array(obs_node(p0, th_e, st_e)[1]) 
end


function create_nn()
    return Chain(
        SkipConnection(Dense(1,3), (out, u) -> u*out[1].+u.*u.*out[2].+u.*log.(abs.(u)).*out[3]),
    )
end

# Define a callback function to observe training
callback = function (p, l, pred; doplot = true)
    l_l = length(lhist)
    println("Loss[$(l_l)]: $(l)")
    push!(lhist, l)
    if doplot
        # plot rolling average of loss, every 10 steps
        if l_l%10 == 0
            plot()
            fig = plot(; xlabel = "Iterations", title = "Loss")
            plot!(fig, 1:10:length(lhist), [mean(lhist[i:min(i+9, length(lhist))]) for i in 1:10:length(lhist)], label = "")
            display(fig)
        end
    end
    return false
end
