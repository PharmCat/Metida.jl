#linearalgebra.jl

# use dot(a,b,a) instead
#=
"""
a' * B * a
"""
function mulαtβα(a::AbstractVector, B::AbstractMatrix{T}) where T
    if length(a) != size(B, 2)::Int  || size(B, 2)::Int  != size(B, 1)::Int  error("Dimention error") end
    axbm  = axes(B, 1)
    axbn  = axes(B, 2)
    c = zero(T)
    for i ∈ axbm
        ct = zero(T)
        for j ∈ axbn
            @inbounds  ct += B[i, j] * a[j]
        end
        @inbounds c += ct * a[i]
    end
    c
end
=#
# Fine
"""
θ + A * B * A'

Change θ (only upper triangle). B is symmetric.
"""
function mulαβαtinc!(θ::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix)
    axb  = axes(B, 1)
    sa   = size(A, 1)
    @simd  for j ∈ axb
        @simd for i ∈ axb
            Bij = B[i, j]
            @simd  for n ∈ 1:sa
                Anj = A[n, j]
                @simd for m ∈ 1:n
                    @inbounds θ[m, n] +=  A[m, i] * Bij * Anj
                end
            end
        end
    end
    θ
end
#=
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
=#
"""
θ + A * B * A' * alpha

Change θ (only upper triangle). B is symmetric.
"""
function mulαβαtinc!(θ::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, alpha)
    if  !(size(B, 1) == size(B, 2) == size(A, 2)) || !(size(A, 1) == size(θ, 1) == size(θ, 2)) throw(ArgumentError("Wrong dimentions!")) end
    axb  = axes(B, 1)
    sa   = size(A, 1)
    @simd for j ∈ axb
        @simd for i ∈ axb
            @inbounds Bij = B[i, j]
            @simd  for n ∈ 1:sa 
                @inbounds Anj = A[n, j]
                @simd for m ∈ 1:n
                    @inbounds θ[m, n] +=  A[m, i] * Bij * Anj * alpha
                end
            end
        end
    end
    θ
end
#=
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
=#
"""
θ + A * B * (a - b) * alpha

Change θ (only upper triangle). B is symmetric.
"""
function mulαβαtinc!(θ::AbstractVector{T}, A::AbstractMatrix, B::AbstractMatrix, a::AbstractVector, b::AbstractVector, alpha) where T
    if !(size(B, 2) == length(a) == length(b)) || size(B, 1) != size(A, 2) || size(A, 1) != length(θ) throw(ArgumentError("Wrong dimentions.")) end
    axb  = axes(B, 1)
    sa   = size(A, 1)
    @simd for i ∈ axb
        abi = a[i] - b[i]
        @simd for j ∈ axb
            @simd for m ∈ 1:sa
                @inbounds θ[m] +=  A[m, j] * B[j, i] * abi * alpha
            end
        end
    end
    θ
end
#=
function mulαβαtinc!(θ::AbstractVector, A::AbstractMatrix, B::AbstractMatrix, a::AbstractVector, b::AbstractVector, alpha)
    if !(size(B, 2) == length(a) == length(b)) || size(B, 1) != size(A, 2) || size(A, 1) != length(θ) throw(ArgumentError("Wrong dimentions.")) end
    axb  = axes(B, 1)
    sa   = size(A, 1)
    for m ∈ 1:sa
        for j ∈ axb
            Amj = A[m, j]
            for i ∈ axb
                @inbounds θ[m] +=  Amj * B[j, i] * (a[i] - b[i]) * alpha
            end
        end
    end
    θ
end
=#
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
        for m in 1:p
            @inbounds cs += X[1, m] * β[m]
        end
        return -V[1, 1] * (y[1] - cs)^2
    end
    c = zeros(T, q)
    @simd for m = 1:p
        βm = β[m]
        @simd for n = 1:q
            @inbounds c[n] += X[n, m] * βm
        end
    end
    @simd for m = 2:q
        ycm = y[m] - c[m]
        @simd for n = 1:m-1
            @inbounds θ -= V[n, m] * (y[n] - c[n]) * ycm * 2
        end
    end
    @simd for m = 1:q
        @inbounds θ -= V[m, m] * (y[m] - c[m]) ^ 2
    end
    return θ
end
#=
function mulθ₃(y, X, β, V::AbstractArray{T}) where T # check for optimization
    q = size(V, 1)
    p = size(X, 2)
    θ = zero(T)

    if q == 1
        cs = zero(T)
        @inbounds  for m in 1:p
            cs += X[1, m] * β[m]
        end
        return -V[1, 1] * (y[1] - cs)^2
    end

    c = zeros(T, q)
    for n = 1:q
        for m = 1:p
            c[n] += X[n, m] * β[m]
        end
    end

    @simd for n = 1:q-1
        ycn = y[n] - c[n]
        @simd for m = n+1:q
            @inbounds θ -= V[n, m] * ycn * (y[m] - c[m]) * 2
        end
    end
    @inbounds  for m = 1:q
        θ -= V[m, m] * (y[m] - c[m]) ^ 2
    end

    return θ
end
=#
"""
θ + A' * b

Change θ.
"""
function mulαtβinc!(θ::AbstractVector{T}, A::AbstractMatrix, b::AbstractVector) where T
    q = size(A, 1)
    if q != length(b) throw(DimensionMismatch("size(A, 1) should be equal length(b)")) end
    p = size(A, 2)
    for n in 1:p
        θn = zero(T)
        for m in 1:q
            @inbounds θn += b[m] * A[m, n]
        end
        @inbounds θ[n] += θn
    end
    θ
end
################################################################################


@inline function tmul_unsafe(rz, θ::AbstractVector{T}) where T
    vec = zeros(T, size(rz, 1))
    for i ∈ axes(rz, 2)
        θi = θ[i]
        for r ∈ axes(rz, 1)
            @inbounds vec[r] += rz[r, i] * θi
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
