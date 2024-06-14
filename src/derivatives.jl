# Derivatives using finite differences

function first_derivatives(u, Δx, Δy = 0.0f0, Δz = 0.0f0)
    println("Warning: this function is not optimized, so you should not use it!")
    dims = ndims(u) - 1 # Subtract 1 for the batch dimension
    if dims == 1
        du_dx = similar(u)
        du_dx[2:(end - 1), :] = (u[3:end, :] - u[1:(end - 2), :])
        du_dx[1, :] = (u[2, :] - u[end, :])
        du_dx[end, :] = (u[1, :] - u[end - 1, :])
        return du_dx / (2 * Δx)
        # TODO add : for sample dimension for 2D and 3D
    elseif dims == 2
        du_dx = similar(u)
        du_dy = similar(u)

        du_dx[2:(end - 1), :] = (u[3:end, :] - u[1:(end - 2), :]) / (2 * Δx)
        du_dx[1, :] = (u[2, :] - u[end, :]) / (2 * Δx)
        du_dx[end, :] = (u[1, :] - u[end - 1, :]) / (2 * Δx)

        du_dy[:, 2:(end - 1)] = (u[:, 3:end] - u[:, 1:(end - 2)]) / (2 * Δy)
        du_dy[:, 1] = (u[:, 2] - u[:, end]) / (2 * Δy)
        du_dy[:, end] = (u[:, 1] - u[:, end - 1]) / (2 * Δy)

        return du_dx, du_dy
    elseif dims == 3
        du_dx = similar(u)
        du_dy = similar(u)
        du_dz = similar(u)

        du_dx[2:(end - 1), :, :] = (u[3:end, :, :] - u[1:(end - 2), :, :]) / (2 * Δx)
        du_dx[1, :, :] = (u[2, :, :] - u[end, :, :]) / (2 * Δx)
        du_dx[end, :, :] = (u[1, :, :] - u[end - 1, :, :]) / (2 * Δx)

        du_dy[:, 2:(end - 1), :] = (u[:, 3:end, :] - u[:, 1:(end - 2), :]) / (2 * Δy)
        du_dy[:, 1, :] = (u[:, 2, :] - u[:, end, :]) / (2 * Δy)
        du_dy[:, end, :] = (u[:, 1, :] - u[:, end - 1, :]) / (2 * Δy)

        du_dz[:, :, 2:(end - 1)] = (u[:, :, 3:end] - u[:, :, 1:(end - 2)]) / (2 * Δz)
        du_dz[:, :, 1] = (u[:, :, 2] - u[:, :, end]) / (2 * Δz)
        du_dz[:, :, end] = (u[:, :, 1] - u[:, :, end - 1]) / (2 * Δz)

        return du_dx, du_dy, du_dz
    else
        error("Unsupported number of dimensions: $dims")
    end
end

function Laplacian(u, Δx2, Δy2 = 0.0, Δz2 = 0.0)
    dims = ndims(u) - 1  # Subtract 1 for the batch dimension

    if dims == 2
        upx = ShiftedArrays.circshift(u, (1, 0, 0))
        upy = ShiftedArrays.circshift(u, (0, 1, 0))
        umx = ShiftedArrays.circshift(u, (-1, 0, 0))
        umy = ShiftedArrays.circshift(u, (0, -1, 0))
        println(typeof(upx))
        println(typeof(umx))
        println(typeof(upy))
        println(typeof(umy))
        println(typeof(u))
        println(typeof(Δx2))
        println(typeof(Δy2))

        #@. (upx + umx + upy + umy - u * 4) / (Δx2 + Δy2)
        @. (upx + u * 4) / (Δx2 + Δy2)
    elseif dims == 1
        up = ShiftedArrays.circshift(u, 1)
        um = ShiftedArrays.circshift(u, -1)

        @. (up + um - u * 2) / Δx2
    else
        error("Unsupported number of dimensions: $dims")
    end
end
