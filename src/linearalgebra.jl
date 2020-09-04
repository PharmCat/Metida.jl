#linearalgebra.jl
"""
A * B * A' + C
"""
function mulαβαtc(A, B, C)
    q  = size(B, 1)
    p  = size(A, 1)
    c  = zeros(eltype(B), q)
    mx = zeros(eltype(B), p, p)
    for i = 1:p
        c .= 0
        @simd for n = 1:q
            @simd for m = 1:q
                @inbounds c[n] +=  A[i, m] * B[n, m]
            end
        end
        @simd for n = 1:p
            @simd for m = 1:q
                 @inbounds mx[i, n] += A[n, m] * c[m]
            end
        end
    end
    mx .+= C
end

#-------------------------------------------------------------------------------
"""
A' * B * A -> + θ
"""
function mulαtβαinc!(θ, A, B)
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

"""
A * B * A -> C
"""
function mulαβαc!(C, A, B)
end

"""
tr(A * B)
"""
function trmulαβ(A, B)
end
"""
tr(H * A' * B * A)
"""
function trmulhαtβα(H, A, B)
end



"""
(y - X * β)' * V * (y - X * β)
"""
function mulθ₃(y::AbstractVector, X::AbstractMatrix, β::AbstractVector, V::AbstractMatrix)
    q = size(V, 1)
    p = size(X, 2)
    θ = 0
    c = zeros(eltype(V), q)
    @simd for n = 1:q
        @simd for m = 1:p
            @inbounds c[n] += X[n, m] * β[m]
        end
    end
    @simd for n = 1:q
        @simd for m = 1:q
            @inbounds θ += V[m, n] * (y[m] - c[m]) * (y[n] - c[n])
        end
    end
    return θ
end
