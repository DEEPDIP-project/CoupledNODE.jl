##################
# This code defines the a priori loss function.
# [!] Observation:
# It is possible that you can directly measure the commutator error, if your problem allows it.
# In this code, the commutator error is not directly measured, but the closure term is trained to reproduce the right-hand side of the equation.
# This means that you could save time by directly training the NN to reproduce the commutator error, if it is available (TODO???). 

import Zygote
import Random: shuffle
import LinearAlgebra: norm

"""
    mean_squared_error(f, st, x, y, θ, λ)

Random a priori loss function. Use this function to train the closure term to reproduce the right hand side.

# Arguments:
- `f`: Function that represents the model.
- `st`: State of the model.
- `x`: Input data.
- `y`: Target data.
- `θ`: Parameters of the model.
- `λ`: Regularization parameter.

# Returns:
- `total_loss`: Mean squared error loss.
"""
# TODO: update docstring
function mean_squared_error(f, st, x, y, θ, λ, λ_c)
    prediction = Array(f(x, θ, st)[1])
    total_loss = sum(abs2, prediction - y) / sum(abs2, y)
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
    create_randloss_derivative(GS_data, FG_target, f, st; nuse = size(GS_data, 2), λ=0)

Create a randomized loss function that compares the derivatives.
This function creates a randomized loss function derivative by selecting a subset of the data. This is done because using the entire dataset at each iteration would be too expensive.

# Arguments
- `GS_data`: The input data.
- `FG_target`: The target data.
- `f`: The model function.
- `st`: The model state.
- `nuse`: The number of samples to use for the loss function. Defaults to the size of `GS_data`.
- `λ`: The regularization parameter. Defaults to 0.

# Returns
A function `randloss` that computes the mean squared error loss using the selected subset of data.
"""
# TODO: update docstring
function create_randloss_derivative(input_data,
        F_target,
        f,
        st;
        nuse = size(input_data, 2),
        λ = 0,
        λ_c = 0)
    d = ndims(input_data)
    nsample = size(input_data, d)
    function randloss(θ)
        i = Zygote.@ignore sort(shuffle(1:nsample)[1:nuse])
        x_use = Zygote.@ignore ArrayType(selectdim(input_data, d, i))
        y_use = Zygote.@ignore ArrayType(selectdim(F_target, d, i))
        mean_squared_error(f, st, x_use, y_use, θ, λ, λ_c)
    end
end
