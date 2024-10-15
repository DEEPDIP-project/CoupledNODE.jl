using CairoMakie: CairoMakie
using Statistics: mean
using CoupledNODE.NavierStokes: create_dataloader_posteriori, create_dataloader_prior

"""
    create_callback(model, val_io_data; lhist=[], lhist_train=[], nunroll=10, rng=rng, plot_train=true)

Create a callback function for training and validation of a model.

# Arguments
- `model`: The model for the rhs.
- `val_io_data`: The validation input-output data for validation.
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
        model, θ, val_io_data, loss_function, st;
        callbackstate = (;
            θmin = θ, loss_min = eltype(θ)(Inf), lhist_val = [], lhist_train = []),
        nunroll = nothing, batch_size = nothing, rng = rng, do_plot = true,
        plot_train = true, plot_every = 10)
    if nunroll === nothing
        if batch_size === nothing
            error("Either nunroll or batch_size must be provided")
        else
            @info "Creating a priori callback"
            dataloader = create_dataloader_prior(val_io_data; batchsize = batch_size, rng)
        end
    else
        @info "Creating a posteriori callback"
        dataloader = create_dataloader_posteriori(val_io_data; nunroll = nunroll, rng)
    end
    # select a fixed sample for the validation
    y1, y2 = dataloader()
    no_model_loss = loss_function(model, θ .* 0, st, (y1, y2))[1]

    function callback(p, l_train)
        step = length(callbackstate.lhist_val)
        # to compute the validation loss, use the parameters p at this step
        l_val = loss_function(model, p, st, (y1, y2))[1]

        @info "Training Loss[$(step)]: $(l_train)"
        @info "Validation Loss[$(step)]: $(l_val)"
        push!(callbackstate.lhist_val, l_val)
        push!(callbackstate.lhist_train, l_train)
        l_val < callbackstate.loss_min &&
            (callbackstate = (; callbackstate..., θmin = θ, loss_min = l_val))
        if do_plot
            fig = CairoMakie.Figure()
            ax = CairoMakie.Axis(fig[1, 1], title = "Loss", xlabel = "Iterations",
                ylabel = "Loss", yscale = CairoMakie.log10)
            # plot rolling average of loss, every plot_every steps
            if step % plot_every == 0
                x = 1:plot_every:length(callbackstate.lhist_val)
                y = [mean(callbackstate.lhist_val[i:min(
                         i + plot_every - 1, length(callbackstate.lhist_val))])
                     for i in 1:plot_every:length(callbackstate.lhist_val)]
                CairoMakie.lines!(ax, x, y, label = "Validation")
                if plot_train
                    x = 1:plot_every:length(callbackstate.lhist_train)
                    y = [mean(callbackstate.lhist_train[i:min(
                             i + plot_every - 1, length(callbackstate.lhist_train))])
                         for i in 1:plot_every:length(callbackstate.lhist_train)]
                    CairoMakie.lines!(ax, x, y, label = "Training")
                end
                CairoMakie.lines!(ax, [0, length(callbackstate.lhist_val)],
                    [no_model_loss, no_model_loss], label = "Val (no closure)")
                CairoMakie.axislegend(ax)
                display(fig)
            end
        end
        callbackstate
    end
    (; callbackstate, callback)
end
