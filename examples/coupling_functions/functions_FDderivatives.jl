# Derivatives using finite differences

function first_derivatives(u, Δx, Δy)
    du_dx = zeros(size(u))
    du_dy = zeros(size(u))

    du_dx[:, 2:end-1] = (u[:, 3:end] - u[:, 1:end-2]) / (2 * Δx)
    du_dx[:, 1] = (u[:, 2] - u[:, end]) / (2 * Δx)
    du_dx[:, end] = (u[:, 1] - u[:, end-1]) / (2 * Δx)

    du_dy[2:end-1, :] = (u[3:end, :] - u[1:end-2, :]) / (2 * Δy)
    du_dy[1, :] = (u[2, :] - u[end, :]) / (2 * Δy)
    du_dy[end, :] = (u[1, :] - u[end-1, :]) / (2 * Δy)

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
    # I use concatenation to avoid in-place operations
    
    # Compute second derivative with respect to x
    d2u_dx2_middle = (u[3:end, :, :] - 2*u[2:end-1, :, :] + u[1:end-2, :, :]) / (Δx^2)
    d2u_dx2_first = (u[2, :, :] - 2*u[1, :, :] + u[end, :, :]) / (Δx^2)
    d2u_dx2_last = (u[1, :, :] - 2*u[end, :, :] + u[end-1, :, :]) / (Δx^2)
    # add a dimension of size 1 to the first and last dimension
    d2u_dx2_first = reshape(d2u_dx2_first, 1, size(d2u_dx2_first)...)
    d2u_dx2_last = reshape(d2u_dx2_last, 1, size(d2u_dx2_last)...)
    # Concatenate
    d2u_dx2 = cat(d2u_dx2_first, d2u_dx2_middle, d2u_dx2_last, dims=1)

    # Compute second derivative with respect to y
    d2u_dy2_middle = (u[:, 3:end, :] - 2*u[:, 2:end-1, :] + u[:, 1:end-2, :]) / (Δy^2)
    d2u_dy2_first = (u[:, 2, :] - 2*u[:, 1, :] + u[:, end, :]) / (Δy^2)
    d2u_dy2_last = (u[:, 1, :] - 2*u[:, end, :] + u[:, end-1, :]) / (Δy^2)
    # add a dimension of size 1 to the first and last dimension
    d2u_dy2_first = reshape(d2u_dy2_first, size(d2u_dy2_first,1),1, size(d2u_dy2_first,2))
    d2u_dy2_last = reshape(d2u_dy2_last, size(d2u_dy2_last,1),1, size(d2u_dy2_last,2))
    # Concatenate
    d2u_dy2 = cat(d2u_dy2_first, d2u_dy2_middle, d2u_dy2_last, dims=2)

    return d2u_dx2, d2u_dy2
end


function Laplacian(u, Δx, Δy)
    d2u_dx2, d2u_dy2 = second_derivatives(u, Δx, Δy)
    return d2u_dx2 + d2u_dy2
end
    