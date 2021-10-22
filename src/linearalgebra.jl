#linearalgebra.jl
"""
a' * B * a
"""
function mulαtβα(a::AbstractVector, B::AbstractMatrix{T}) where T
    if length(a) != size(B, 2)::Int  || size(B, 2)::Int  != size(B, 1)::Int  error("Dimention error") end
    c = zero(T)
    for i ∈ axes(B, 1)::Base.OneTo{Int}
        ct = zero(T)
        for i2 ∈ axes(B, 2)::Base.OneTo{Int}
            @inbounds  ct += B[i, i2]*a[i2]
        end
        @inbounds c += ct*a[i]
    end
    c
end
"""
θ + A * B * A'

Change θ (only upper triangle).
"""
function mulαβαtinc!(θ::AbstractMatrix{T}, A::AbstractMatrix, B::AbstractMatrix) where T
    #1 alloc
    axb  = axes(B, 1)
    axa  = axes(A, 1)
    sa   = size(A, 1)
    c  = Vector{T}(undef, size(B, 1))
    for i ∈ axa
        fill!(c, zero(T))
        @inbounds  for n ∈ axb, m ∈ axb
            c[n] += B[m, n] * A[i, m]
        end
        #upper triangular n = i:p
        @inbounds  for n ∈ i:sa, m ∈ axb
            θ[i, n] += A[n, m] * c[m]
        end
    end
    θ
end
"""
    mulθ₃(y, X, β, V::AbstractMatrix{T})::T where T

(y - X * β)' * (-V) * (y - X * β)

use only upper triangle of V
"""
function mulθ₃(y, X, β, V::AbstractArray{T}) where T
    q = size(V, 1)
    p = size(X, 2)
    θ = zero(T)

    if q == 1
        c = zero(T)
        #=@turbo=# @inbounds  for m in 1:p
            c += X[1, m] * β[m]
        end
        return -V[1, 1] * (y[1] - c)^2
    end

    c = zeros(T, q)
    #=@turbo=# @inbounds  for n = 1:q, m = 1:p
        c[n] += X[n, m] * β[m]
    end

    @simd for n = 1:q
        @simd for m = n+1:q
            @inbounds θ -= V[n, m] * (y[n] - c[n]) * (y[m] - c[m]) * 2
        end
    end
    #=@turbo=# @inbounds  for m = 1:q
        θ -= V[m, m] * (y[m] - c[m]) ^ 2
    end

    return θ
end
"""
θ + A' * b

Change θ.
"""
function mulαtβinc!(θ::AbstractVector, A::AbstractMatrix, b::AbstractVector)
    q = size(A, 1)
    if q != length(b) throw(DimensionMismatch("size(A, 1) should be equal length(b)")) end
    p = size(A, 2)
    #=@turbo=# @inbounds  for n in 1:p, m in 1:q
        θ[n] += b[m] * A[m, n]
    end
    θ
end
################################################################################

#=
If length θ >= size(rz, 2)
!!! without checking
vec = rz * θ
=#
@inline function tmul_unsafe(rz, θ::AbstractVector{T}) where T
    vec = zeros(T, size(rz, 1))
    #=@turbo=# @inbounds  for r ∈ axes(rz, 1)
        for i ∈ axes(rz, 2)
            vec[r] += rz[r, i] * θ[i]
        end
    end
    vec
end
