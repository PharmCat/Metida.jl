#linearalgebra.jl

# Fine
"""
    mulαβαtinc!(θ::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix)

θ + A * B * A'

Change θ (only upper triangle). B is symmetric.
"""
@noinline function mulαβαtinc!(θ::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix)
    axb  = axes(B, 1)
    sa   = size(A, 1)
    for j ∈ axb
        for i ∈ axb
            @inbounds Bij = B[i, j]
           for n ∈ 1:sa
                @inbounds Anj = A[n, j]
                BijAnj = Bij * Anj
                @simd for m ∈ 1:n
                    @inbounds θ[m, n] +=  A[m, i] * BijAnj
                end
            end
        end
    end
    return θ
end
#=
function mulαβαtinc!(θ::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where T <: AbstractFloat
    if  !(size(B, 1) == size(B, 2) == size(A, 2)) || !(size(A, 1) == size(θ, 1) == size(θ, 2)) throw(ArgumentError("Wrong dimentions!")) end
    t = A*B
    mul!(θ, t, A', true, true)
end
=#
"""
    mulαβαtinc!(θ::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, alpha)

θ + A * B * A' * alpha

Change θ (only upper triangle). B is symmetric.
"""
@noinline function mulαβαtinc!(θ::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, alpha)
    if  !(size(B, 1) == size(B, 2) == size(A, 2)) || !(size(A, 1) == size(θ, 1) == size(θ, 2)) throw(ArgumentError("Wrong dimentions!")) end
    axb  = axes(B, 1)
    sa   = size(A, 1)
   for j ∈ axb
        for i ∈ axb
            @inbounds Bij = B[i, j]
            for n ∈ 1:sa 
                @inbounds Anj = A[n, j]
                BijAnjalpha = Bij * Anj * alpha
                @simd for m ∈ 1:n
                    @inbounds θ[m, n] +=  A[m, i] * BijAnjalpha
                end
            end
        end
    end
    return θ
end
"""
    mulαβαtinc!(θ::AbstractVector{T}, A::AbstractMatrix, B::AbstractMatrix, a::AbstractVector, b::AbstractVector, alpha) where T

θ + A * B * (a - b) * alpha

Change θ (only upper triangle). B is symmetric.
"""
@noinline function mulαβαtinc!(θ::AbstractVector{T}, A::AbstractMatrix, B::AbstractMatrix, a::AbstractVector, b::AbstractVector, alpha) where T
    if !(size(B, 2) == length(a) == length(b)) || size(B, 1) != size(A, 2) || size(A, 1) != length(θ) throw(ArgumentError("Wrong dimentions.")) end
    axb  = axes(B, 1)
    sa   = size(A, 1)
    for i ∈ axb
        @inbounds abi = a[i] - b[i]
       for j ∈ axb
            @inbounds Bji = B[j, i]
            Bjiabialpha = Bji * abi * alpha
            @simd for m ∈ 1:sa
                @inbounds θ[m] +=  A[m, j] * Bjiabialpha
            end
        end
    end
    return θ
end

"""
    mulθ₃(y, X, β, V::AbstractMatrix{T})::T where T

(y - X * β)' * (-V) * (y - X * β)

use only upper triangle of V
"""
@noinline function mulθ₃(y, X, β, V::AbstractArray{T}) where T # check for optimization
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
    for m = 1:p
        @inbounds βm = β[m]
        @simd for n = 1:q
            @inbounds c[n] += X[n, m] * βm
        end
    end
    for m = 2:q
        @inbounds ycm2 = (y[m] - c[m]) * 2
        @simd for n = 1:m - 1
            @inbounds θ -= V[n, m] * (y[n] - c[n]) * ycm2
        end
    end
    @simd for m = 1:q
        @inbounds θ -= V[m, m] * (y[m] - c[m]) ^ 2
    end
    return θ
end

"""
    mulαtβinc!(θ::AbstractVector{T}, A::AbstractMatrix, b::AbstractVector) where T

θ + A' * b

Change θ.
"""
@noinline function mulαtβinc!(θ::AbstractVector{T}, A::AbstractMatrix, b::AbstractVector) where T
    q = size(A, 1)
    if q != length(b) throw(DimensionMismatch("size(A, 1) should be equal length(b)")) end
    p = size(A, 2)
    for n in 1:p
        θn = zero(T)
        @simd for m in 1:q
            @inbounds θn += b[m] * A[m, n]
        end
        @inbounds θ[n] += θn
    end
    return θ
end
# Diagonal(b) * A * Diagonal(b) - chnage only A upper triangle 
@noinline function mulβdαβd!(A::AbstractMatrix, b::AbstractVector)
    q = size(A, 1)
    p = size(A, 2)
    if !(q == p == length(b)) throw(DimensionMismatch("size(A, 1) and size(A, 2) should be equal length(b)")) end
    for n in 1:p
        @simd for m in 1:n
            @inbounds A[m, n] *= b[m] * b[n]
        end
    end
    return A
end


################################################################################
@inline function tmul_unsafe(rz, θ::AbstractVector{T}) where T
    vec = zeros(T, size(rz, 1))
    for i ∈ axes(rz, 2)
        @inbounds θi = θ[i]
        @simd for r ∈ axes(rz, 1)
            @inbounds vec[r] += rz[r, i] * θi
        end
    end
    return vec
end

@inline function diag!(f, v, m)
    l = checksquare(m)
    l == length(v) || error("Length not equal")
    @simd for i = 1:l
        @inbounds v[i] = f(m[i, i])
    end
    return v
end
