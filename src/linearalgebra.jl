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
            @inbounds  ct += B[i, i2] * a[i2]
        end
        @inbounds c += ct * a[i]
    end
    c
end
"""
θ + A * B * A'

Change θ (only upper triangle). B is symmetric.
"""
function mulαβαtinc!(θ::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix)
    axb  = axes(B, 1)
    sa   = size(A, 1)
    for m ∈ 1:sa
        for n ∈ m:sa
            for j ∈ axb
                @inbounds for i ∈ axb
                    θ[m, n] +=  A[m, i] * B[i, j] * A[n, j]
                end
            end
        end
    end
    θ
end
"""
θ + A * B * A' * alpha

Change θ (only upper triangle). B is symmetric.
"""
function mulαβαtinc!(θ::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, alpha)
    if  !(size(B, 1) == size(B, 2) == size(A, 2)) || !(size(A, 1) == size(θ, 1) == size(θ, 2)) throw(ArgumentError("Wrong dimentions!")) end
    axb  = axes(B, 1)
    sa   = size(A, 1)
    for m ∈ 1:sa
        for n ∈ m:sa
            for j ∈ axb
                for i ∈ axb
                    @inbounds  θ[m, n] +=  A[m, i] * B[i, j] * A[n, j] * alpha
                end
            end
        end
    end
    θ
end
"""
θ + A * B * (a - b) * alpha

Change θ (only upper triangle). B is symmetric.
"""
function mulαβαtinc!(θ::AbstractVector, A::AbstractMatrix, B::AbstractMatrix, a::AbstractVector, b::AbstractVector, alpha)
    if !(size(B, 2) == length(a) == length(b)) || size(B, 1) != size(A, 2) || size(A, 1) != length(θ) throw(ArgumentError("Wrong dimentions.")) end
    axb  = axes(B, 1)
    sa   = size(A, 1)
    for m ∈ 1:sa
        for j ∈ axb
            for i ∈ axb
                @inbounds θ[m] +=  A[m, j] * B[j, i] * (a[i] - b[i]) * alpha
            end
        end
    end
    θ
end
"""
    mulθ₃(y, X, β, V::AbstractMatrix{T})::T where T

(y - X * β)' * (-V) * (y - X * β)

use only upper triangle of V
"""
function mulθ₃(y, X, β, V::AbstractArray{T}) where T # check for optimization
    q = size(V, 1)
    p = size(X, 2)
    θ = zero(T)

    if q == 1
        cs = zero(T)
        #=@turbo=# @inbounds  for m in 1:p
            cs += X[1, m] * β[m]
        end
        return -V[1, 1] * (y[1] - cs)^2
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
    #=@turbo=# @inbounds  for m in 1:q, n in 1:p
        θ[n] += b[m] * A[m, n]
    end
    θ
end
################################################################################


@inline function tmul_unsafe(rz, θ::AbstractVector{T}) where T
    vec = zeros(T, size(rz, 1))
    #=@turbo=# for r ∈ axes(rz, 1)
        for i ∈ axes(rz, 2)
            @inbounds vec[r] += rz[r, i] * θ[i]
        end
    end
    vec
end

function diag!(f, v, m)
    l = checksquare(m)
    l == length(v) || error("Length not equal")
    for i = 1:l
        v[i] = f(m[i, i])
    end
    v
end
