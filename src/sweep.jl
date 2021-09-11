#sweep.jl
#Based on https://github.com/joshday/SweepOperator.jl
#Thanks to @joshday and @Hua-Zhou

function nsyrk!(alpha, A, C)
    q = size(C, 1)
    #p = size(A, 2)
    @simd for n ∈ axes(C, 1)
        @simd for m ∈ n:q
            @inbounds @simd for i ∈ axes(A, 2)
                C[n, m] += A[n, i] * A[m, i] * alpha
            end
        end
    end
    C
end
function sweep!(A::AbstractArray{T}, k::Integer, inv::Bool = false; syrkblas::Bool = false) where T
    sweepb!(Vector{T}(undef, size(A, 2)), A, k, inv; syrkblas = syrkblas)
end
function sweepb!(akk::AbstractArray{T, 1}, A::AbstractArray{T, 2}, k::Integer, inv::Bool = false; syrkblas::Bool = false) where T <: Number
    p = checksquare(A)
    #p == length(akk) || throw(DimensionError("incorrect buffer size"))
    @inbounds d = one(T) / A[k, k]
    @simd for j in 1:k
        @inbounds akk[j] = A[j, k]
    end
    @simd for j in (k+1):p
        @inbounds akk[j] = A[k, j]
    end
    #syrk!(uplo, trans, alpha, A, beta, C)
    #Rank-k update of the symmetric matrix C as alpha*A*transpose(A) + beta*C
    #or alpha*transpose(A)*A + beta*C according to trans.
    #Only the uplo triangle of C is used. Returns C.
    if syrkblas
        BLAS.syrk!('U', 'N', -d, akk, one(T), A)
    else
        nsyrk!(-d, akk, A)
    end
    rmul!(akk, d * (-one(T)) ^ inv)
    @simd for i in 1:(k-1)
        @inbounds A[i, k] = akk[i]
    end
    @simd for j in (k+1):p
        @inbounds A[k, j] = akk[j]
    end
    @inbounds A[k, k] = -d
    A
end
function sweep!(A::AbstractArray{T, 2}, ks::AbstractVector{I}, inv::Bool = false; syrkblas::Bool = false, logdet::Bool = false) where {T <: Number, I <: Integer}
    akk = Vector{T}(undef, size(A,2))
    sweepb!(akk, A, ks, inv; syrkblas = syrkblas, logdet = logdet)
end
function sweepb!(akk::AbstractArray{T, 1}, A::AbstractArray{T, 2}, ks::AbstractVector{I}, inv::Bool = false; syrkblas::Bool = false, logdet::Bool = false) where
        {T <: Number, I<:Integer}
    ld = NaN
    noerror = true
    if logdet
        ld = 0
        @inbounds for k in ks
            Akk = A[k,k]
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
            sweepb!(akk, A, k, inv; syrkblas = syrkblas)
        end
    else
        for k in ks
            sweepb!(akk, A, k, inv; syrkblas = syrkblas)
        end
    end
    A, ld, noerror
end
