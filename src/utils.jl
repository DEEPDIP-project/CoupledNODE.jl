using CairoMakie: CairoMakie
using Statistics: mean
using CoupledNODE.NavierStokes: create_dataloader_posteriori, create_dataloader_prior

function create_stateful_callback(
        θ,
        err_function = nothing,
        callbackstate = (; θmin = θ, θmin_e = θ, loss_min = eltype(θ)(Inf),
            emin = eltype(θ)(Inf), hist = CairoMakie.Point2f[]),
        displayref = false,
        display_each_iteration = true,
        filename = nothing
)
    #istart = isempty(callbackstate.hist) ? 0 : Int(callbackstate.hist[end][1])
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

"""
    create_callback(model, test_io_data; lhist=[], lhist_train=[], nunroll=10, rng=rng, plot_train=true)

Create a callback function for training and validation of a model.

# Arguments
- `model`: The model for the rhs.
- `test_io_data`: The test input-output data for validation.
- `lhist`: A list to store the history of validation losses. Defaults to a new empty list.
- `lhist_train`: A list to store the history of training losses. Defaults to a new empty list.
- `nunroll`: The number of unroll steps for the validation loss. It does not have to be the same as the loss function!
- `rng`: The random number generator to be used. Defaults to `rng`.
- `plot_train`: A boolean flag to indicate whether to plot the training loss.

# Returns
A callback function that can be used during training to compute and log validation and training losses, and optionally plot the loss history.

# Callback Function Arguments
- `p`: The parameters of the model at the current training step.
- `ltrain`: The training loss at the current training step.
- `pred`: Optional. The predictions of the model at the current training step. Defaults to `nothing`.
- `do_plot`: Optional. A boolean flag to indicate whether to plot the loss history. Defaults to `true`.
- `return_lhist`: Optional. A boolean flag to indicate whether to return the loss history. Defaults to `false`.

# Callback Function Returns
- If `return_lhist` is `true`, returns the validation and training loss histories (`lhist` and `lhist_train`), without computing anything else.
- Otherwise, returns `false` to operate as expected from a callback function.
"""
function create_callback(
        model, test_io_data, loss_function, st; lhist = [], lhist_train = [],
        nunroll = nothing, batch_size = nothing, rng = rng, do_plot = true, plot_train = true)
    if nunroll === nothing
        if batch_size === nothing
            error("Either nunroll or batch_size must be provided")
        else
            print("Creating a priori callback")
            dataloader = create_dataloader_prior(test_io_data; batchsize = batch_size, rng)
        end
    else
        print("Creating a posteriori callback")
        dataloader = create_dataloader_posteriori(test_io_data; nunroll = nunroll, rng)
    end
    # select a fixed sample for the validation
    y1, y2 = dataloader()
    no_model_loss = nothing

    function (p, ltrain, pred = nothing; return_lhist = false)
        if return_lhist
            return lhist, lhist_train
        end
        l_l = length(lhist)
        if no_model_loss === nothing
            # reference loss without model
            no_model_loss = loss_function(model, p .* 0, st, (y1, y2))[1]
        end
        # to compute the validation loss, use the parameters p at this step
        l = loss_function(model, p, st, (y1, y2))[1]

        @info "Training Loss[$(l_l)]: $(ltrain)"
        @info "Validation Loss[$(l_l)]: $(l)"
        push!(lhist, l)
        push!(lhist_train, ltrain)
        if do_plot
            fig = CairoMakie.Figure()
            ax = CairoMakie.Axis(fig[1, 1], title = "Loss", xlabel = "Iterations",
                ylabel = "Loss", yscale = CairoMakie.log10)
            # plot rolling average of loss, every 10 steps
            if l_l % 10 == 0
                x = 1:10:length(lhist)
                y = [mean(lhist[i:min(i + 9, length(lhist))]) for i in 1:10:length(lhist)]
                CairoMakie.lines!(ax, x, y, label = "Validation")
                if plot_train
                    x = 1:10:length(lhist_train)
                    y = [mean(lhist[i:min(i + 9, length(lhist_train))])
                         for i in 1:10:length(lhist_train)]
                    CairoMakie.lines!(ax, x, y, label = "Training")
                end
                CairoMakie.lines!(ax, [0, length(lhist)], [no_model_loss, no_model_loss],
                    label = "Val (no closure)")
                CairoMakie.axislegend(ax)
                display(fig)
            end
        end
        return false
    end
end
