#sweep.jl
#Based on https://github.com/joshday/SweepOperator.jl
#Thanks to @joshday and @Hua-Zhou

function nsyrk!(α, x, A)
    p = checksquare(A)
    for j in 1:p
        @inbounds xjα = x[j] * α
        @simd for i in 1:j 
            @inbounds A[i, j] += x[i] * xjα
        end
    end
    return A
end
function nsyrk!(α, x, A::AbstractArray{T}) where T <: AbstractFloat
    return BLAS.syrk!('U', 'N', α, x, one(T), A)
end

function sweep!(A::AbstractArray{T}, k::Integer, inv::Bool = false) where T
    return sweepb!(Vector{T}(undef, size(A, 2)), A, k, inv)
end
function sweepb!(akk::AbstractArray{T, 1}, A::AbstractArray{T, 2}, k::Integer, inv::Bool = false) where T <: Number
    p = checksquare(A)
    #p == length(akk) || throw(DimensionError("incorrect buffer size"))
    @inbounds d = one(T) / A[k, k]
    @simd for j in 1:k
        @inbounds akk[j] = A[j, k]
    end
    @simd for j in (k + 1):p
        @inbounds akk[j] = A[k, j]
    end
    # syrk!(uplo, trans, alpha, A, beta, C)
    # Rank-k update of the symmetric matrix C as alpha*A*transpose(A) + beta*C
    # or alpha*transpose(A)*A + beta*C according to trans.
    # Only the uplo triangle of C is used. Returns C.

    nsyrk!(-d, akk, A)
    
    rmul!(akk, d * (-one(T)) ^ inv)
    @simd for i in 1:(k-1)
        @inbounds A[i, k] = akk[i]
    end
    @simd for j in (k+1):p
        @inbounds A[k, j] = akk[j]
    end
    @inbounds A[k, k] = -d
    return A
end
function sweep!(A::AbstractArray{T, 2}, ks::AbstractVector{I}, inv::Bool = false; logdet::Bool = false) where {T <: Number, I <: Integer}
    akk = Vector{T}(undef, size(A,2))
    return sweepb!(akk, A, ks, inv; logdet = logdet)
end
function sweepb!(akk::AbstractArray{T, 1}, A::AbstractArray{T, 2}, ks::AbstractVector{I}, inv::Bool = false; logdet::Bool = false) where
        {T <: Number, I<:Integer}
    ld = NaN
    noerror = true
    if logdet
        ld = 0
        for k in ks
            @inbounds Akk = A[k, k]
            if Akk > 0
                ld += log(Akk)
            else
                noerror = false
                if Akk < 0
                    ld += log(- Akk * LDCORR)
                else
                    ld += LOGLDCORR
                end
            end
            sweepb!(akk, A, k, inv)
        end
    else
        for k in ks
            sweepb!(akk, A, k, inv)
        end
    end
    return A, ld, noerror
end
