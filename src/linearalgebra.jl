#linearalgebra.jl
#=
"""
```math
    \\begin{bmatrix} A * B * A' & X \\\\ X' & 0 \\end{bmatrix}
```
"""
function mulαβαt3(A, B, X)
    q  = size(B, 1)
    p  = size(A, 1)
    c  = zeros(eltype(B), q)
    mx = zeros(eltype(B), p + size(X, 2), p + size(X, 2))
    for i = 1:p
        fill!(c, zero(eltype(c)))
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
    mx[1:p, p+1:end] = X
    mx[p+1:end, 1:p] = X'
    mx
end
=#
"""
θ + A * B * A'

Change θ.
"""
function mulαβαtinc!(θ::AbstractVecOrMat{T}, A, B) where T
    q = size(B, 1)
    p = size(A, 1)
    c = zeros(T, q)
    for i = 1:p
        fill!(c, zero(T))
        @inbounds for n = 1:q, m = 1:q
            c[n] += B[m, n] * A[i, m]
        end
        @inbounds for n = 1:p, m = 1:q
            θ[i, n] += A[n, m] * c[m]
        end
    end
    θ
end
"""
(y - X * β)' * V * (y - X * β)
"""
function mulθ₃(y, X, β, V::AbstractMatrix{T})::T where T
    q = size(V, 1)
    p = size(X, 2)
    θ = zero(T)
    c = zeros(T, q)
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
"""
θ + A' * B

Change θ.
"""
function mulαtβinc!(θ::AbstractVecOrMat{T}, A::AbstractMatrix, B::AbstractVector) where T
    if size(A, 1) != length(B) throw(DimensionMismatch("size(A, 1) should be equal length(B)")) end
    q = size(A, 1)
    p = size(A, 2)
    @simd for n = 1:p
        @simd for m = 1:q
            @inbounds θ[n] += B[m] * A[m, n]
        end
    end
    θ
end

################################################################################
function utriaply!(f, m)
    if size(m, 1) != size(m, 2) error() end
    @simd for p = 1:size(m, 1)
        @simd for q = p:size(m, 2)
            @inbounds m[p, q] = f(m[p, q])
        end
    end
    m
end
