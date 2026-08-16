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



 
# =============================================================================
#  1. Ядра: BLAS-путь и generic-путь (Dual)
# =============================================================================
 
# --- C += Aᵀ A, только верхний треугольник C ---------------------------------
@inline function _syrk_upper!(C::AbstractMatrix{T}, A::AbstractMatrix{T}) where T <: BlasFloat
    BLAS.syrk!('U', 'T', one(T), A, one(T), C)
    return C
end
function _syrk_upper!(C::AbstractMatrix{T}, A::AbstractMatrix) where T
    m, k = size(A, 1), size(A, 2)
    @inbounds for j in 1:k
        for i in 1:j
            s = zero(T)
            @simd for l in 1:m            # оба столбца читаются подряд
                s += A[l, i] * A[l, j]
            end
            C[i, j] += s
        end
    end
    return C
end
 
# --- решение Rᵀ Z = B (R — верхний фактор Холецкого) -------------------------
@inline function _ltsolve!(R::AbstractMatrix{T}, B::AbstractMatrix{T}) where T <: BlasFloat
    BLAS.trsm!('L', 'U', 'T', 'N', one(T), R, B)
    return B
end
@inline function _ltsolve!(R::AbstractMatrix, B::AbstractMatrix)
    ldiv!(UpperTriangular(R)', B)
    return B
end
@inline function _ltsolve!(R::AbstractMatrix{T}, b::AbstractVector{T}) where T <: BlasFloat
    BLAS.trsv!('U', 'T', 'N', R, b)
    return b
end
@inline function _ltsolve!(R::AbstractMatrix, b::AbstractVector)
    ldiv!(UpperTriangular(R)', b)
    return b
end
 
# --- решение R x = b ---------------------------------------------------------
@inline function _usolve!(R::AbstractMatrix{T}, b::AbstractVector{T}) where T <: BlasFloat
    BLAS.trsv!('U', 'N', 'N', R, b)
    return b
end
@inline function _usolve!(R::AbstractMatrix, b::AbstractVector)
    ldiv!(UpperTriangular(R), b)
    return b
end
 
# --- xᵀ A x по верхнему треугольнику A ---------------------------------------
function _qform_upper(A::AbstractMatrix{T}, x::AbstractVector) where T
    p = size(A, 1)
    s = zero(promote_type(T, eltype(x)))
    @inbounds for j in 1:p
        xj = x[j]
        s += A[j, j] * xj * xj
        @simd for i in 1:j-1
            s += 2 * A[i, j] * x[i] * xj
        end
    end
    return s
end
 

