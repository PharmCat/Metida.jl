#linearalgebra.jl
"""
A * B * A' + C
"""
@inline function mulαβαtc(A, B, C)
    q  = size(B, 1)
    p  = size(A, 1)
    c  = zeros(eltype(B), q)
    mx = zeros(eltype(B), p, p)
    for i = 1:p
        fill!(c, zero(eltype(c)))
        @simd for n = 1:q
            @simd for m = 1:q
                @inbounds c[n] +=  A[i, m] * B[n, m]
            end
        end
        @simd for n = i:p
            @simd for m = 1:q
                 @inbounds mx[i, n] += A[n, m] * c[m]
            end
            @inbounds mx[i, n] += C[i, n]
        end
    end
    Symmetric(mx)
end
@inline function mulαβαtc2(A, B, C, r::Vector)
    q  = size(B, 1)
    p  = size(A, 1)
    c  = zeros(eltype(B), q)
    mx = zeros(eltype(B), p + 1, p + 1)
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
            @inbounds mx[i, n] += C[i, n]
        end
        mx[end, i] = r[i]
        mx[i, end] = r[i]
    end
    mx
end
@inline function mulαβαtc3(A, B, C, X)
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
            @inbounds mx[i, n] += C[i, n]
        end
    end
    mx[1:p, p+1:end] = X
    mx[p+1:end, 1:p] = X'
    mx
end
#-------------------------------------------------------------------------------
"""
A' * B * A -> + θ
"""
@inline function mulαtβαinc!(θ, A, B)
    q = size(B, 1)
    p = size(A, 2)
    c = zeros(eltype(B), q)
    for i = 1:p
        fill!(c, zero(eltype(c)))
        @inbounds for n = 1:q, m = 1:q
            c[n] += B[m, n] * A[m, i]
        end
        @inbounds for n = 1:p, m = 1:q
            θ[i, n] += A[m, n] * c[m]
        end
    end
end
"""
A' * B * A -> θ
"""
@inline function mulαtβα!(θ, A, B)
    q = size(B, 1)
    p = size(A, 2)
    c = zeros(eltype(B), q)
    fill!(θ, zero(eltype(θ)))
    for i = 1:p
        fill!(c, zero(eltype(c)))
        @inbounds for n = 1:q, m = 1:q
            c[n] += B[m, n] * A[m, i]
        end
        @inbounds for n = 1:p, m = 1:q
            θ[i, n] += A[m, n] * c[m]
        end
    end
    θ
end
"""
A * B * A -> θ
"""
@inline function mulαβαc!(θ, A, B)
    q = size(B, 1)
    p = size(A, 2)
    c = zeros(eltype(B), q)
    fill!(θ, zero(eltype(θ)))
    for i = 1:p
        fill!(c, zero(eltype(c)))
        @inbounds for n = 1:q, m = 1:q
            c[n] += B[m, n] * A[i, m]
        end
        @inbounds for n = 1:p, m = 1:q
            θ[i, n] += A[m, n] * c[m]
        end
    end
    θ
end
"""
tr(A * B)
"""
@inline function trmulαβ(A, B)
    c = 0
    @inbounds for n = 1:size(A,1), m = 1:size(B, 1)
        c += A[n,m] * B[m, n]
    end
    c
end
"""
tr(H * A' * B * A)
"""
function trmulhαtβα(H, A, B)
end
"""
(y - X * β)' * V * (y - X * β)
"""
@inline function mulθ₃(y::AbstractVector, X::AbstractMatrix, β::AbstractVector, V::AbstractMatrix{T})::T where T
    q = size(V, 1)
    p = size(X, 2)
    θ = zero(T)
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
"""
(y - X * β)
"""
@inline function mulr!(v::AbstractVector, y::AbstractVector, X::AbstractMatrix, β::AbstractVector)
    fill!(v, zero(eltype(v)))
    q = length(y)
    p = size(X, 2)
    @simd for n = 1:q
        @simd for m = 1:p
            @inbounds v[n] += X[n, m] * β[m]
        end
        v[n] = y[n] - v[n]
    end
    return v
end
@inline function mulr(y::AbstractVector, X::AbstractMatrix, β::AbstractVector)
    v = zeros(eltype(β), length(y))
    q = length(y)
    p = size(X, 2)
    @simd for n = 1:q
        @simd for m = 1:p
            @inbounds v[n] += X[n, m] * β[m]
        end
        v[n] = y[n] - v[n]
    end
    return v
end
"""
A' * B  -> + θ
"""
@inline function mulαtβinc!(θ, A::AbstractMatrix, B::AbstractVector)
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


#=
d = 0.4
akk = [1.0 2.0 4.0 ; 6.0 1.0 3.0; 3 4  5]
b = 0.3
A = [2.0 5.0 2.3; 6.0 6.0 4.5 ; 1 2 3]
BLAS.syrk!('U', 'N', -d, akk, b, A)
-d*akk*akk'+b*A
nsyrk!(-d, akk, b, A)
=#
