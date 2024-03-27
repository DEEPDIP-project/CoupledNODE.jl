import Statistics: mean

using Plots

"""
    callback(p, l, pred; doplot=true)

The callback function is used to observe training progress. It prints the current loss value and updates the rolling average of the loss. If `do_plot` is `true`, it also plots the rolling average of the loss every 10 steps.

# Arguments
- `p`: the current parameters
- `l`: the current loss value
- `pred`: the current prediction
- `do_plot`: a boolean indicating whether to plot the rolling average of the loss every 10 steps (default is `true`)

# Returns
- `false` if nothing unexpected happened.
"""
lhist = [] # do not know if defining this outside is correct julia
callback = function (p, l, pred; do_plot = true)
    global lhist
    l_l = length(lhist)
    println("Loss[$(l_l)]: $(l)")
    push!(lhist, l)
    if do_plot
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
