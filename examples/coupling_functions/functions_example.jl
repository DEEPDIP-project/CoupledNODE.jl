using DifferentialEquations
using Plots

function observation()
    f_o(u, p, t) = u .* (0.0 .- 0.8 .* log.(u))
    trange = (0.0, 6.0)
    prob = ODEProblem(f_o, 0.01, trange, dt=0.01, saveat=0.01)
    sol = solve(prob, Tsit5())
    return sol.u
end

# Define a callback function to observe training
callback = function (p, l, pred; doplot = true)
    l_l = length(lhist)
    println("Loss[$(l_l)]: $(l)")
    push!(lhist, l)
    if doplot
        # plot rolling average of loss, every 10 steps
        if l_l % 10 == 0
            plot()
            fig = plot(; xlabel = "Iterations", title = "Loss", yscale = :log10)
            plot!(fig,
                1:10:length(lhist),
                [mean(lhist[i:min(i + 9, length(lhist))]) for i in 1:10:length(lhist)],
                label = "")
            display(fig)
        end
    end
    return false
end
