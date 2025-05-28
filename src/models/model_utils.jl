"""
Interpolate velocity components to volume centers.

TODO, D and dir can be parameters istead of arguments I think
"""
function interpolate(A, D, dir)
    (i, a) = A
    if i > D
        return a  # Nothing to interpolate for extra layers
    end
    staggered = a .+ circshift(a, ntuple(x -> x == i ? dir : 0, D))
    staggered ./ 2
end

function collocate(u)
    D = ndims(u) - 2
    slices = eachslice(u; dims = D + 1)
    staggered_slices = map(x -> interpolate(x, D, 1), enumerate(slices))
    stack(staggered_slices; dims = D + 1)
end

"""
Interpolate closure force from volume centers to volume faces.
"""
function decollocate(u)
    D = ndims(u) - 2
    slices = eachslice(u; dims = D + 1)
    staggered_slices = map(x -> interpolate(x, D, -1), enumerate(slices))
    stack(staggered_slices; dims = D + 1)
end
