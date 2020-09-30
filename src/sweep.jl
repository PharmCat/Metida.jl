#sweep.jl
#Based on https://github.com/joshday/SweepOperator.jl
#Thanks to @joshday and @Hua-Zhou

function nsyrk!(alpha, A, C)
    q = size(C, 1)
    p = size(A, 2)
    for n = 1:q
        for m = n:q
            c = zero(eltype(C))
            for i = 1:p
                @inbounds c += A[n, i] * A[m, i]
            end
            @inbounds C[n, m] = C[n, m] + c * alpha
        end
    end
    C
end

function sweep!(A::AbstractMatrix, k::Integer, inv::Bool = false)
    sweepb!(Vector{eltype(A)}(undef, size(A, 2)), A, k, inv)
end
function sweepb!(akk::AbstractVector{T}, A::AbstractMatrix{T}, k::Integer, inv::Bool = false) where T <: Number
    p = checksquare(A)
    p == length(akk) || throw(DimensionError("incorrect buffer size"))
    @inbounds d = one(T) / A[k, k]
    @simd for j in 1:k
        @inbounds akk[j] = A[j, k]
    end
    @simd for j in (k+1):p
        @inbounds akk[j] = A[k, j]
    end
    nsyrk!(-d, akk, A)
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
function sweep!(A::AbstractMatrix{T}, ks::AbstractVector{I}, inv::Bool = false) where
    {T <: Number, I <: Integer}
    akk = zeros(T, size(A, 1))
    for k in ks
        sweepb!(akk, A, k, inv)
    end
    A
end
function sweepb!(akk::AbstractVector{T}, A::AbstractMatrix{T}, ks::AbstractVector{I}, inv::Bool = false) where
        {T <: Number, I<:Integer}
    for k in ks
        sweepb!(akk, A, k, inv)
    end
    A
end
