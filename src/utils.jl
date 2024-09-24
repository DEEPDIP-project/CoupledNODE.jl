using CairoMakie: CairoMakie
using Plots: Plots
using Statistics: mean

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
callback = function (p, l, pred = nothing; do_plot = true)
    global lhist
    l_l = length(lhist)
    @info "Loss[$(l_l)]: $(l)"
    push!(lhist, l)
    if do_plot
        # plot rolling average of loss, every 10 steps
        if l_l % 10 == 0
            Plots.plot()
            fig = Plots.plot(; xlabel = "Iterations", title = "Loss", yscale = :log10)
            Plots.plot!(fig,
                1:10:length(lhist),
                [mean(lhist[i:min(i + 9, length(lhist))]) for i in 1:10:length(lhist)],
                label = "")
            display(fig)
        end
    end
    return false
end

function create_stateful_callback(
        θ,
        err_function = nothing,
        callbackstate = (; θmin = θ, θmin_e = θ, loss_min = eltype(θ)(Inf),
            emin = eltype(θ)(Inf), hist = CairoMakie.Point2f[]),
        displayref = false,
        display_each_iteration = true,
        filename = nothing
)
    istart = isempty(callbackstate.hist) ? 0 : Int(callbackstate.hist[end][1])
    obs = CairoMakie.Observable([CairoMakie.Point2f(0, 0)])
    fig = CairoMakie.lines(obs; axis = (; title = "Error", xlabel = "step"))
    displayref && CairoMakie.hlines!([1.0f0]; linestyle = :dash)
    obs[] = callbackstate.hist
    display(fig)
    function callback(θ, loss)
        if err_function !== nothing
            e = err_function(θ)
            #@info "Iteration $i \terror: $e"
            e < state.emin && (callbackstate = (; callbackstate..., θmin_e = θ, emin = e))
        end
        hist = push!(
            copy(callbackstate.hist), CairoMakie.Point2f(length(callbackstate.hist), loss))
        obs[] = hist
        CairoMakie.autolimits!(fig.axis)
        display_each_iteration && display(fig)
        isnothing(filename) || save(filename, fig)
        callbackstate = (; callbackstate..., hist)
        loss < callbackstate.loss_min &&
            (callbackstate = (; callbackstate..., θmin = θ, loss_min = loss))
        callbackstate
    end
    (; callbackstate, callback)
end
