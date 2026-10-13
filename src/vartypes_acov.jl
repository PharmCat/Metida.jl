################################################################################
#                      ACOV   (covariance modifiers)
################################################################################
# A modifier reads the R block built by the PRECEDING repeated effects and
# augments (ACOV) it. `corr` returns the correlation applied to
# cell (m, n); the variances already on the diagonal supply the scale.
#
# `ct.a` only takes effect on a cell that is already NONZERO, as documented:
#   0 - add            1 - replace        2 - keep the existing value
#   3 - add, warn      4 - replace, warn  5 - keep, warn
# Warnings are raised once per model at construction time, not per cell.

struct ACOV_{C <: AbstractCovarianceType} <: AbstractCovarianceType
    c::C
    a::Int
end

"""
    ACOV(c, action = 0)

!!! warning
    Experimental
    
Augmented (adjusted) covariance. Add additional correlations to existed R-part of variance covariance matrix.
Can be used with `AR` or `CS` types. 

`action` if existed covariance not equal sero:

* 0 - add
* 1 - replace
* 2 - do nothing (use existed value)
* 3 - warning and add
* 4 - warning and replace
* 5 - warning and use existed value
* other - error

"""
function ACOV(c; action = 0)
    CovarianceType(ACOV_(c.s, action))
end

function covstrparam(ct::ACOV_{<:Union{CS_, AR_, SPPOW_}}, ::Int)::Tuple{Int, Int}
    return (0, 1)
end

function rcoefnames(s, t, ct::ACOV_{<: Union{CS_, AR_, SPPOW_}})
    return ["ρ "]
end

function Base.show(io::IO, ct::ACOV_)
    print(io, "ACOV(", ct.c, ")")
end

@inline function acov_apply!(mx::AbstractMatrix{T}, ct::ACOV_, corr::F) where {T, F}
    s = size(mx, 1)
    s > 1 || return mx
    a = ct.a
    @inbounds for n = 2:s
        mxnn = mx[n, n]
        for m = 1:n-1
            mxmn = mx[m, n]
            cov  = sqrt(mxnn * mx[m, m]) * corr(m, n)
            if !iszero(mxmn)                    # documented guard
                if a == 1 || a == 4
                    mxmn = zero(T)              # replace
                elseif a == 2 || a == 5
                    cov  = zero(T)              # keep existing
                elseif a > 5
                    error("ACOV: covariance cell ($m, $n) already has a nonzero value (action = $a).")
                end
            end
            mx[m, n] = mxmn + cov
        end
    end
    return mx
end

# ACOV (AR): lag is the row-position difference inside the block
function rmat!(mx::AbstractMatrix{T}, θ, ::AbstractMatrix, ct::ACOV_{AR_}, ::Int) where T
    ρ = θ[1]
    return acov_apply!(mx, ct, (m, n) -> ρ ^ (n - m))
end

# ACOV (CS): exchangeable, order free
function rmat!(mx::AbstractMatrix{T}, θ, ::AbstractMatrix, ct::ACOV_{CS_}, ::Int) where T
    ρ = θ[1]
    return acov_apply!(mx, ct, (m, n) -> ρ)
end

# ACOV (SPPOW): lag is the Euclidean distance between the rows of `rz`, so the
# correlation follows the actual spacing of a time or coordinate variable and
# does not depend on how the rows happen to be ordered.
function rmat!(mx::AbstractMatrix{T}, θ, rz::AbstractMatrix, ct::ACOV_{SPPOW_}, ::Int) where T
    ρ = θ[1]
    return acov_apply!(mx, ct, (m, n) -> ρ ^ edistance(rz, m, n))
end



################################################################################
#                       BAND  (covariance modifiers)
################################################################################
# A modifier reads the R block built by the PRECEDING repeated effects and
# masks (BAND) it. 

struct BAND_ <: AbstractCovarianceType
    k::Int
end

"""
    BAND(k)

!!! warning
    Experimental

Band modificator. Zeroes every covariance cell whose within-block lag exceeds
`k`, leaving the diagonal and the first `k` lags untouched. Contributes no
parameters. Used as a subsequent repeated effect, after the structure it masks.

A band is not block diagonal, so it cannot be reproduced by a blocking factor;
masking by a factor, in contrast, is already obtained for free by giving that
effect the narrower blocking factor.
"""
function BAND(k::Int)
    k >= 0 || throw(ArgumentError("BAND: k must be non-negative, got $k"))
    CovarianceType(BAND_(k))
end

function covstrparam(ct::BAND_, ::Int)::Tuple{Int, Int}
    return (0, 0)
end

function rcoefnames(s, t, ct::BAND_)
    return String[]
end

function Base.show(io::IO, ct::BAND_)
    print(io, "BAND(", ct.k, ")")
end

# BAND(k): zero every cell with lag > k. Parameter free.
function rmat!(mx::AbstractMatrix{T}, θ, ::AbstractMatrix, ct::BAND_, ::Int) where T
    s = size(mx, 1)
    k = ct.k
    @inbounds for n = 2:s
        for m = 1:n-1
            if n - m > k
                mx[m, n] = zero(T)
            end
        end
    end
    return mx
end