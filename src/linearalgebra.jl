#linearalgebra.jl

#-------------------------------------------------------------------------------
"""
A' * B * A -> + θ
"""
function mulαtβαinc!(θ, A::AbstractMatrix, B::AbstractMatrix)
    q = size(B, 1)
    p = size(A, 2)
    c = zeros(eltype(B), q)
    for i = 1:p
        c .= 0
        @inbounds for n = 1:q, m = 1:q
            c[n] += B[m, n] * A[m, i]
        end
        @inbounds for n = 1:p, m = 1:q
            θ[i, n] += A[m, n] * c[m]
        end
    end
    #θ
end
