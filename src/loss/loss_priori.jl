# It is possible to directly measure the commutator error (during data generation), if your problem allows it.
# In the 2 first functions, the commutator error is not directly measured, but the closure term is trained to reproduce the right-hand side of the equation.
# This means that you could save time by directly training the NN to reproduce the commutator error, if it is available. -> Last two methods in this file.

using Zygote: Zygote
using Random: shuffle
using LinearAlgebra: norm
using Lux: Lux

"""
    mean_squared_error(f, st, x, y, θ, λ)

Random a priori loss function. Use this function to train the closure term to reproduce the right hand side.

# Arguments:
- `f`: Function that represents the model.
- `st`: State of the model.
- `x`: Input data.
- `y`: Target data.
- `θ`: Parameters of the model.
- `λ` and `λ_c`: Regularization parameters.

# Returns:
- `total_loss`: Mean squared error loss.
"""
function mean_squared_error(f, st, x, y, θ, λ, λ_c; dim = 1)
    prediction = Array(f(x, θ, st)[1])
    total_loss = 0
    if dim > 1
        for i in 1:length(x)
            total_loss += sum(abs2, prediction[i] - y[i])
        end
    else
        total_loss = sum(abs2, prediction - y) / sum(abs2, y)
    end
    # add regularization term
    if λ > 0
        normterm = norm(θ, 1)
    else
        normterm = 0
    end
    # add a continuity term that penalises force with strong discontinuities
    if λ_c > 0
        first_last_diff = sum(abs.(prediction[end, :] - prediction[1, :]))
        cterm = sum(abs.(diff(prediction, dims = 1))) + first_last_diff
    else
        cterm = 0
    end
    return total_loss, nothing
end

"""
    create_randloss_derivative(input_data, F_target, f, st; n_use = size(input_data, 2), λ=0, λ_c = 0)

Create a randomized loss function that compares the derivatives.
This is done because using the entire dataset at each iteration would be too expensive.

# Arguments
- `input_data`: The input data.
- `F_target`: The target data.
- `f`: The model function.
- `st`: The model state.
- `n_use`: The number of samples to use for the loss function. Defaults to the number of samples in the input data.
- `λ` and `λ_c`: The regularization parameter. Defaults to 0.

# Returns
A function that computes the mean squared error loss and takes as input the model parameters.
"""
function create_randloss_derivative(
        input_data, F_target, f, st; dim = 1, n_use = 64, λ = 0, λ_c = 0)
    # [!] dim is the number of fields and not the field dimension (e.g is ~[u,v] not ~[ux, uy])
    if dim == 1
        sd = length(size(input_data))
        n_samples = size(input_data)[end]
        return θ -> begin
            i = Zygote.@ignore sort(shuffle(1:n_samples)[1:n_use])
            x_use = Zygote.@ignore selectdim(input_data, sd, i)
            y_use = Zygote.@ignore selectdim(F_target, sd, i)
            mean_squared_error(f, st, x_use, y_use, θ, λ, λ_c)
        end
    elseif dim == 2
        x1, x2 = input_data.x
        y1, y2 = F_target.x
        sd = length(size(x1))
        n_samples = size(x1)[end]
        return θ -> begin
            i = Zygote.@ignore sort(shuffle(1:n_samples)[1:n_use])
            x1_use = Zygote.@ignore selectdim(x1, sd, i)
            x2_use = Zygote.@ignore selectdim(x2, sd, i)
            y1_use = Zygote.@ignore selectdim(y1, sd, i)
            y2_use = Zygote.@ignore selectdim(y2, sd, i)
            mean_squared_error(f, st, ArrayPartition(x1_use, x2_use),
                ArrayPartition(y1_use, y2_use), θ, λ, λ_c, dim = dim)
        end
    else
        error("ERROR: Unsupported number of dimensions: $dim")
    end
end

"""
Wrap loss function `loss(batch, θ)`.

The function `loss` should take inputs like `loss(f, x, y, θ)`.
"""
create_loss_priori(loss, f) = (f, θ, st, (x, y)) -> loss(f, θ, st, (x, y))

"""
Compute MSE between `f(x, θ, st)` and `y`.
The MSE is further divided by `normalize(y)`.
This signature has been chosen for compatibility with Lux.
"""
mean_squared_error(f, θ, st, (x, y); normalize = y -> sum(abs2, y), λ = sqrt(1e-8)) = sum(
    abs2, f(x, θ, st)[1] - y) / normalize(y) + eltype(x)(λ) *
                                                                                      sum(
    abs2, θ)

"""
    loss_priori_lux(model, ps, st, (x, y))

Compute the loss  as the sum of squared differences between the predicted values and the target values, normalized by the sum of squared target values.
Format of function signature and outputs are compatible with the Lux ecosystem.

# Arguments
- `model`: The model to be evaluated.
- `ps`: Parameters for the model.
- `st`: State of the model.
- `(x, y)`: A tuple where `x` is the input data and `y` is the target data.

# Returns
- `loss`: The computed loss value.
- `st_`: The updated state of the model.
- `(; y_pred = y_pred)`: Named tuple containing the predicted values `y_pred`.
"""
function loss_priori_lux(model, ps, st, (x, y), device=identity)
    y_pred, st_ = model(x, ps, st)
    loss = device(sum(abs2, y_pred .- y) / sum(abs2, y))
    return loss, st_, (; y_pred = y_pred)
end
