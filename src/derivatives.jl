# Derivatives using finite differences

function first_derivatives(u, Δx, Δy)
    du_dx = zeros(size(u))
    du_dy = zeros(size(u))

    du_dx[:, 2:(end - 1)] = (u[:, 3:end] - u[:, 1:(end - 2)]) / (2 * Δx)
    du_dx[:, 1] = (u[:, 2] - u[:, end]) / (2 * Δx)
    du_dx[:, end] = (u[:, 1] - u[:, end - 1]) / (2 * Δx)

    du_dy[2:(end - 1), :] = (u[3:end, :] - u[1:(end - 2), :]) / (2 * Δy)
    du_dy[1, :] = (u[2, :] - u[end, :]) / (2 * Δy)
    du_dy[end, :] = (u[1, :] - u[end - 1, :]) / (2 * Δy)

    return du_dx, du_dy
end
#function second_derivatives(u, Δx, Δy)
#    d2u_dx2 = zeros(size(u))
#    d2u_dy2 = zeros(size(u))
#
#    d2u_dx2[:, 2:end-1, :] = (u[:, 3:end, :] - 2*u[:, 2:end-1, :] + u[:, 1:end-2, :]) / (Δx^2)
#    d2u_dx2[:, 1, :] = (u[:, 2, :] - 2*u[:, 1, :] + u[:, end, :]) / (Δx^2)
#    d2u_dx2[:, end, :] = (u[:, 1, :] - 2*u[:, end, :] + u[:, end-1, :]) / (Δx^2)
#
#    d2u_dy2[2:end-1, :, :] = (u[3:end, :, :] - 2*u[2:end-1, :, :] + u[1:end-2, :, :]) / (Δy^2)
#    d2u_dy2[1, :, :] = (u[2, :, :] - 2*u[1, :, :] + u[end, :, :]) / (Δy^2)
#    d2u_dy2[end, :, :] = (u[1, :, :] - 2*u[end, :, :] + u[end-1, :, :]) / (Δy^2)
#
#    return d2u_dx2, d2u_dy2
#end
function second_derivatives(u, Δx, Δy)
    # Use concatenation to avoid in-place operations
    # Compute second derivative with respect to x
    d2u_dx2_middle = (u[3:end, :, :] - 2 * u[2:(end - 1), :, :] + u[1:(end - 2), :, :]) /
                     (Δx^2)
    d2u_dx2_first = (u[2, :, :] - 2 * u[1, :, :] + u[end, :, :]) / (Δx^2)
    d2u_dx2_last = (u[1, :, :] - 2 * u[end, :, :] + u[end - 1, :, :]) / (Δx^2)
    # add a dimension of size 1 to the first and last dimension
    d2u_dx2_first = reshape(d2u_dx2_first, 1, size(d2u_dx2_first)...)
    d2u_dx2_last = reshape(d2u_dx2_last, 1, size(d2u_dx2_last)...)
    # Concatenate
    d2u_dx2 = cat(d2u_dx2_first, d2u_dx2_middle, d2u_dx2_last, dims = 1)

    # Compute second derivative with respect to y
    d2u_dy2_middle = (u[:, 3:end, :] - 2 * u[:, 2:(end - 1), :] + u[:, 1:(end - 2), :]) /
                     (Δy^2)
    d2u_dy2_first = (u[:, 2, :] - 2 * u[:, 1, :] + u[:, end, :]) / (Δy^2)
    d2u_dy2_last = (u[:, 1, :] - 2 * u[:, end, :] + u[:, end - 1, :]) / (Δy^2)
    # add a dimension of size 1 to the first and last dimension
    d2u_dy2_first = reshape(d2u_dy2_first,
        size(d2u_dy2_first, 1),
        1,
        size(d2u_dy2_first, 2))
    d2u_dy2_last = reshape(d2u_dy2_last, size(d2u_dy2_last, 1), 1, size(d2u_dy2_last, 2))
    # Concatenate
    d2u_dy2 = cat(d2u_dy2_first, d2u_dy2_middle, d2u_dy2_last, dims = 2)

    return d2u_dx2, d2u_dy2
end

function Laplacian_old(u, Δx, Δy)
    d2u_dx2, d2u_dy2 = second_derivatives(u, Δx, Δy)
    return d2u_dx2 + d2u_dy2
end

function circular_pad(u)
    add_dim_1(x) = reshape(x, 1, size(x)...)
    add_dim_2(x) = reshape(x, size(x, 1), 1, size(x, 2))
    u_padded = vcat(add_dim_1(u[end, :, :]), u, add_dim_1(u[1, :, :]))
    u_padded = hcat(add_dim_2(u_padded[:, end, :]), u_padded, add_dim_2(u_padded[:, 1, :]))
    return u_padded
end

function Laplacian(u, Δx2, Δy2)
    up = circular_pad(u)
    d2u = similar(up)

    d2u[2:(end - 1), :, :] = (up[3:end, :, :] - 2 * up[2:(end - 1), :, :] +
                              up[1:(end - 2), :, :])
    d2u[:, 2:(end - 1), :] += (up[:, 3:end, :] - 2 * up[:, 2:(end - 1), :] +
                               up[:, 1:(end - 2), :])

    return d2u[2:(end - 1), 2:(end - 1), :] / (Δx2 + Δy2)
end
