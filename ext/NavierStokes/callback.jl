using CairoMakie: CairoMakie
using Random: Random
using Statistics: mean
using Adapt: adapt

"""
    create_callback(model, val_io_data; lhist=[], lhist_train=[], nunroll=10, rng=rng, plot_train=true)

Create a callback function for training and validation of a model.

# Arguments
- `model`: The model for the rhs.
- `θ`: parameters of the model (trainable).
- `val_io_data`: The validation io_array.
- `loss_function`: The loss function to be used.
- `st`: The state of the model.
- `callbackstate`: a `NamedTuple` that is updated durign the trainign and contains:
    - `θmin`: The parameters at the minimum validation loss.
    - `loss_min`: The minimum validation loss.
    - `lhist_val`: A list to store the history of validation losses. Defaults to a new empty list.
    - `lhist_train`: A list to store the history of training losses. Defaults to a new empty list.
    - `lhist_nomodel`: A list to store the history of the validation loss without the model. Defaults to a new empty list.
- `nunroll`: The number of unroll steps for the validation loss. It does not have to be the same as the loss function! Pertinent for a-posteriori training.
- `rng`: The random number generator to be used.
- `plot_train`: A boolean flag to indicate whether to plot the training loss.
- `do_plot`: A boolean flag to indicate whether to generate the plots. In HPC systems we may want to deactivate it.
- `plot_every`: The frequency of plotting the loss history. Defaults to 10. The loss is also averaged in this window.

# Returns
A `NamedTuple` with the `callbackstate`` and the callback function.
The callback function is used during training to compute and log validation and training losses, and optionally plot the loss history.

# Callback function arguments
- `p`: The parameters of the model at the current training step.
- `l_train`: The training loss at the current training step.

# Callback function returns
- `callbackstate`: the updated instance of the callback state.
"""
function create_callback(
        model, θ, val_io_data, loss_function, st;
        callbackstate = nothing,
        nunroll = nothing,
        batch_size = nothing,
        rng = Random.Xoshiro(123),
        do_plot = true,
        plot_train = true,
        plot_every = 10,
        average_window = 25,
        device = identity,
        figfile = nothing)
    if callbackstate === nothing
        # Initialize the callback state
        # To store data coming from CUDA device, we have to serialize them to CPU
        callbackstate = (;
            θmin = θ, loss_min = eltype(θ)(Inf), lhist_val = [],
            lhist_train = [], lhist_nomodel = [])
    end
    if nunroll === nothing && batch_size === nothing
        error("Either nunroll or batch_size must be provided")
    elseif nunroll !== nothing
        @info "Creating a posteriori callback"
        dataloader = create_dataloader_posteriori(val_io_data; nunroll = nunroll, rng)
    else
        @info "Creating a priori callback"
        dataloader = create_dataloader_prior(val_io_data; batchsize = batch_size, rng)
    end

    function rolling_average(y)
        return [mean(@view y[max(1, i - average_window):i]) for i in 1:length(y)]
    end

    function callback(p, l_train)
        step = length(callbackstate.lhist_train)

        plot_train && @info "[$(step)] Training Loss: $(l_train)"
        push!(callbackstate.lhist_train, l_train)

        if step % plot_every == 0
            y1, y2 = device(dataloader())
            l_val = loss_function(model, p, st, (y1, y2))[1]
            # check if this set of p produces a lower validation loss
            l_val < callbackstate.loss_min &&
                (callbackstate = (; callbackstate..., θmin = p, loss_min = l_val))
            @info "[$(step)] Validation Loss: $(l_val)"
            no_model_loss = loss_function(model, callbackstate.θmin .* 0, st, (y1, y2))[1]
            @info "[$(step)] Validation Loss (no model): $(no_model_loss)"

            push!(callbackstate.lhist_val, l_val)
            push!(callbackstate.lhist_nomodel, no_model_loss)

            if do_plot
                fig = CairoMakie.Figure()
                ax = CairoMakie.Axis(fig[1, 1], title = "Loss", xlabel = "Iterations",
                    ylabel = "Loss", yscale = CairoMakie.log10)
                x = 1:length(callbackstate.lhist_val)
                y = rolling_average(callbackstate.lhist_val)
                CairoMakie.lines!(ax, (x .- 1) .* plot_every, y, label = "Validation")
                if plot_train
                    x = 1:length(callbackstate.lhist_train)
                    y = rolling_average(callbackstate.lhist_train)
                    CairoMakie.lines!(ax, x .- 1, y, label = "Training")
                end
                x = 1:length(callbackstate.lhist_nomodel)
                y = rolling_average(callbackstate.lhist_nomodel)
                CairoMakie.lines!(
                    ax, (x .- 1) .* plot_every, y, label = "Validation (no closure)")

                CairoMakie.axislegend(ax)
                display(fig)

                if figfile !== nothing
                    CairoMakie.save(figfile, fig)
                end
            end
        end
        callbackstate
    end
    (; callbackstate, callback)
end
