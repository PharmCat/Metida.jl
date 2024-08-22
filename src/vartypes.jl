################################################################################
# COVMAT METHOD
################################################################################

################################################################################

struct SI_ <: AbstractCovarianceType end
mutable struct SWC_{W<:AbstractMatrix, B<:Vector{<:AbstractMatrix}} <: AbstractCovarianceType 
    wtsm::W
    wtsb::B
end
struct DIAG_ <: AbstractCovarianceType end
struct AR_ <: AbstractCovarianceType end
struct ARH_ <: AbstractCovarianceType end
struct CS_ <: AbstractCovarianceType end
struct CSH_ <: AbstractCovarianceType end
struct ARMA_ <: AbstractCovarianceType end
struct TOEP_ <: AbstractCovarianceType end
struct TOEPP_ <: AbstractCovarianceType
    p::Int
end
struct TOEPH_ <: AbstractCovarianceType end
struct TOEPHP_ <: AbstractCovarianceType
    p::Int
end
struct SPEXP_ <: AbstractCovarianceType end
struct SPPOW_ <: AbstractCovarianceType end
struct SPGAU_ <: AbstractCovarianceType end
struct SPEXPD_ <: AbstractCovarianceType end
struct SPPOWD_ <: AbstractCovarianceType end
struct SPGAUD_ <: AbstractCovarianceType end
struct UN_ <: AbstractCovarianceType end
struct ZERO <: AbstractCovarianceType end

################################################################################
#                          COVARIANCE TYPE
################################################################################
"""
    CovarianceType(cm::AbstractCovmatMethod)

Make covariance type with AbstractCovmatMethod.

"""
struct CovarianceType{T <: AbstractCovarianceType}
    s::T
    z::Bool
    function CovarianceType(s::T, z::Bool) where T <: AbstractCovarianceType
        new{T}(s, z)
    end
    function CovarianceType(s::AbstractCovarianceType)
        CovarianceType(s,  true)
    end
end

################################################################################
"""
    ScaledIdentity()

Scaled identity covariance type.

SI = ScaledIdentity()

```math
\\begin{bmatrix} 1 & 0 & 0 \\\\ 0 & 1 & 0 \\\\ 0 & 0 & 1 \\end{bmatrix}\\sigma^{2}
```
"""
function ScaledIdentity()
    CovarianceType(SI_())
end
const SI = ScaledIdentity()

# Experimental
"""
    ScaledWeightedCov(wtsm::AbstractMatrix{T})

!!! warning
    Experimental

Scaled weighted covariance matrix, where `wtsm` - `NxN` within block correlation matrix (N - total number of observations). 
Used only for repeated effect. 

SWC = ScaledWeightedCov

```math
R = Corr(W) * \\sigma_c^2
```

where ``Corr(W)`` - diagonal correlation matrix. 

example:

```julia
matwts = Symmetric(UnitUpperTriangular(rand(size(df0,1), size(df0,1))))
lmm = LMM(@formula(var~sequence+period+formulation), df0;
    repeated = VarEffect(@covstr(1|subject), SWC(matwts)))
fit!(lmm)

```
!!! note

There is no `wtsm` checks for symmetricity or values.

"""
function ScaledWeightedCov(wtsm::AbstractMatrix{T}) where T
    wtsb = Matrix{T}[]
    CovarianceType(SWC_(wtsm, wtsb))
end
const SWC = ScaledWeightedCov

"""
    Diag()

Diagonal covariance type.

DIAG = Diag()

```math
\\begin{bmatrix} \\sigma_a^2 & 0 & 0 \\\\ 0 & \\sigma_b^2 & 0 \\\\ 0 & 0 & \\sigma_c^2 \\end{bmatrix}
```
"""
function Diag()
    CovarianceType(DIAG_())
end
const DIAG = Diag()
"""
    Autoregressive()

Autoregressive covariance type.

AR = Autoregressive()

```math
\\begin{bmatrix} 1 & \\rho & \\rho^2 & \\rho^3 \\\\
\\rho & 1 & \\rho & \\rho^2 \\\\ \\rho^2 & \\rho & 1 & \\rho \\\\
\\rho^3 & \\rho^2 & \\rho & 1
\\end{bmatrix}\\sigma^2
```
"""
function Autoregressive()
    CovarianceType(AR_())
end
const AR = Autoregressive()
"""
    HeterogeneousAutoregressive()

Heterogeneous autoregressive covariance type.

ARH = HeterogeneousAutoregressive()

```math
\\begin{bmatrix}
\\sigma_a^2 & \\rho\\sigma_a\\sigma_b & \\rho^2\\sigma_a\\sigma_c & \\rho^3\\sigma_a\\sigma_d \\\\
\\rho\\sigma_b\\sigma_a & \\sigma_b^2 & \\rho\\sigma_b\\sigma_c & \\rho^2\\sigma_b\\sigma_d \\\\
\\rho^2\\sigma_c\\sigma_a & \\rho\\sigma_c\\sigma_b & \\sigma_c^2 & \\rho\\sigma_c\\sigma_d \\\\
\\rho^3\\sigma_d\\sigma_a & \\rho^2\\sigma_d\\sigma_b & \\rho\\sigma_d\\sigma_c & \\sigma_d^2
\\end{bmatrix}
```
"""
function HeterogeneousAutoregressive()
    CovarianceType(ARH_())
end
const ARH = HeterogeneousAutoregressive()
"""
    CompoundSymmetry()

Compound symmetry covariance type.

CS = CompoundSymmetry()

```math
\\begin{bmatrix} 1 & \\rho & \\rho & \\rho \\\\
\\rho & 1 & \\rho & \\rho \\\\
\\rho & \\rho & 1 & \\rho \\\\
\\rho & \\rho & \\rho & 1
\\end{bmatrix}\\sigma^2
```
"""
function CompoundSymmetry()
    CovarianceType(CS_())
end
const CS = CompoundSymmetry()
"""
    HeterogeneousCompoundSymmetry()

Heterogeneous compound symmetry covariance type.

CSH = HeterogeneousCompoundSymmetry()

```math
\\begin{bmatrix}
\\sigma_a^2 & \\rho\\sigma_a\\sigma_b & \\rho\\sigma_a\\sigma_c & \\rho\\sigma_a\\sigma_d \\\\
\\rho\\sigma_b\\sigma_a & \\sigma_b^2 & \\rho\\sigma_b\\sigma_c & \\rho\\sigma_b\\sigma_d \\\\
\\rho\\sigma_c\\sigma_a & \\rho\\sigma_c\\sigma_b & \\sigma_c^2 & \\rho\\sigma_c\\sigma_d \\\\
\\rho\\sigma_d\\sigma_a & \\rho\\sigma_d\\sigma_b & \\rho\\sigma_d\\sigma_c & \\sigma_d^2
\\end{bmatrix}
```
"""
function HeterogeneousCompoundSymmetry()
    CovarianceType(CSH_())
end
const CSH = HeterogeneousCompoundSymmetry()
"""
    AutoregressiveMovingAverage()

Autoregressive moving average covariance type.

ARMA = AutoregressiveMovingAverage()

```math
\\begin{bmatrix} 1 & \\gamma & \\gamma\\rho & \\gamma\\rho^2 \\\\
\\gamma & 1 & \\gamma & \\gamma\\rho \\\\
\\gamma\\rho & \\gamma & 1 & \\gamma \\\\
\\gamma\\rho^2 & \\gamma\\rho & \\gamma & 1
\\end{bmatrix}\\sigma^2
```
"""
function AutoregressiveMovingAverage()
    CovarianceType(ARMA_())
end
const ARMA = AutoregressiveMovingAverage()

"""
    Toeplitz()

Toeplitz covariance type. Only for G matrix.

TOEP = Toeplitz()

```math
\\begin{bmatrix} 1 & \\rho_1 & \\rho_2 & \\rho_3 \\\\
\\rho_1 & 1 & \\rho_1 & \\rho_2 \\\\
\\rho_2 & \\rho_1 & 1 & \\rho_1 \\\\
\\rho_3 & \\rho_2 & \\rho_1 & 1
\\end{bmatrix}\\sigma^2
```

"""
function Toeplitz()
    CovarianceType(TOEP_())
end
const TOEP = Toeplitz()

"""
    ToeplitzParameterized(p::Int)

Toeplitz covariance type with parameter p, (number of bands = p - 1, if p = 1 it's equal SI structure).

TOEPP(p) = ToeplitzParameterized(p)

"""
function ToeplitzParameterized(p::Int)
    CovarianceType(TOEPP_(p))
end
TOEPP(p) = ToeplitzParameterized(p)

"""
    HeterogeneousToeplitz()

Heterogeneous toeplitz covariance type. Only for G matrix.

TOEPH = HeterogeneousToeplitz()

```math
\\begin{bmatrix}
\\sigma_a^2 & \\rho_1 \\sigma_a \\sigma_b & \\rho_2 \\sigma_a \\sigma_c & \\rho_3 \\sigma_a \\sigma_d \\\\
\\rho_1 \\sigma_b \\sigma_a & \\sigma_b^2 & \\rho_1 \\sigma_b \\sigma_c & \\rho_2 \\sigma_b \\sigma_d \\\\
\\rho_2 \\sigma_c \\sigma_a & \\rho_1 \\sigma_c \\sigma_b & \\sigma_c^2 & \\rho_1 \\sigma_c \\sigma_d \\\\
\\rho_3 \\sigma_d \\sigma_a & \\rho_2 \\sigma_d \\sigma_b & \\rho_1 \\sigma_d \\sigma_c & \\sigma_d^2
\\end{bmatrix}
```

"""
function HeterogeneousToeplitz()
    CovarianceType(TOEPH_())
end
const TOEPH = HeterogeneousToeplitz()

"""
    HeterogeneousToeplitzParameterized(p::Int)

Heterogeneous toeplitz covariance type with parameter p, (number of bands = p - 1, if p = 1 it's equal DIAG structure).

TOEPHP(p) = HeterogeneousToeplitzParameterized(p)

"""
function HeterogeneousToeplitzParameterized(p::Int)
    CovarianceType(TOEPHP_(p))
end
TOEPHP(p) = HeterogeneousToeplitzParameterized(p)

"""
    SpatialExponential()

Spatian Exponential covariance structure. Used only for repeated effect.

```math
R_{i,j} = \\sigma^{2} * exp(-dist(i,j)/\\theta)
```

where `dist` - Euclidean distance between row-vectors of repeated effect matrix for subject `i` and `j`, θ > 0.

SPEXP = SpatialExponential()
"""
function SpatialExponential()
    CovarianceType(SPEXP_())
end
const SPEXP = SpatialExponential()

"""
    SpatialPower()

Spatian Power covariance structure. Used only for repeated effect.

```math
R_{i,j} = \\sigma^{2} * \\rho^{dist(i,j)}
```

where `dist` - Euclidean distance between row-vectors of repeated effect matrix for subject `i` and `j`, 1 > ρ > -1.

SPPOW = SpatialPower()
"""
function SpatialPower()
    CovarianceType(SPPOW_())
end
const SPPOW = SpatialPower()

"""
    SpatialGaussian()

Spatian Gaussian covariance structure. Used only for repeated effect.

```math
R_{i,j} = \\sigma^{2} * exp(- dist(i,j)^2 / \\theta^2)
```

where `dist` - Euclidean distance between row-vectors of repeated effect matrix for subject `i` and `j`, θ ≠ 0.

SPGAU = SpatialGaussian()
"""
function SpatialGaussian()
    CovarianceType(SPGAU_())
end
const SPGAU = SpatialGaussian()

"""
    SpatialExponentialD()

!!! warning
    Experimental

Same as SpatialExponential, but add D to all diagonal elements.

SPEXPD = SpatialExponentialD()
"""
function SpatialExponentialD()
    CovarianceType(SPEXPD_())
end
const SPEXPD = SpatialExponentialD()

"""
    SpatialPowerD()

!!! warning
    Experimental

Same as SpatialPower, but add D to all diagonal elements.

SPPOWD = SpatialPowerD()
"""
function SpatialPowerD()
    CovarianceType(SPPOWD_())
end
const SPPOWD = SpatialPowerD()
"""
    SpatialGaussianD()

!!! warning
    Experimental

Same as SpatialGaussianD, but add D to all diagonal elements.

SPGAUD = SpatialGaussianD()
"""
function SpatialGaussianD()
    CovarianceType(SPGAUD_())
end
const SPGAUD = SpatialGaussianD()

"""
    Unstructured()

Unstructured covariance structure with `t*(t+1)/2-t` paremeters where `t` - number of factor levels, `t*(t+1)/2-2t` of them is covariance (ρ) patemeters.

UN = Unstructured()
"""
function Unstructured()
    CovarianceType(UN_())
end
const UN = Unstructured()

function RZero()
    CovarianceType(ZERO(), false)
end

function covstrparam(ct::SI_, ::Int)::Tuple{Int, Int}
    return (1, 0)
end
function covstrparam(ct::SWC_, ::Int)::Tuple{Int, Int}
    return (1, 0)
end
function covstrparam(ct::DIAG_, t::Int, )::Tuple{Int, Int}
    return (t, 0)
end
function covstrparam(ct::Union{AR_, CS_, SPPOW_}, ::Int)::Tuple{Int, Int}
    return (1, 1)
end
function covstrparam(ct::Union{ARH_, CSH_}, t::Int)::Tuple{Int, Int}
    return (t, 1)
end
function covstrparam(ct::ARMA_, ::Int)::Tuple{Int, Int}
    return (1, 2)
end
function covstrparam(ct::TOEP_, t::Int)::Tuple{Int, Int}
    return (1, t - 1)
end
function covstrparam(ct::TOEPH_, t::Int)::Tuple{Int, Int}
    return (t, t - 1)
end
function covstrparam(ct::TOEPP_, ::Int)::Tuple{Int, Int}
    return (1, Int(ct.p - 1))
end
function covstrparam(ct::TOEPHP_, t::Int)::Tuple{Int, Int}
    return (t, Int(ct.p - 1))
end
function covstrparam(ct::UN_, t::Int)::Tuple{Int, Int}
    return (t, Int(t * (t + 1) / 2 - t))
end
function covstrparam(ct::Union{SPEXP_, SPGAU_}, ::Int)::Tuple{Int, Int, Int}
    return (1, 0, 1)
end
function covstrparam(ct::Union{SPEXPD_, SPGAUD_}, ::Int)::Tuple{Int, Int, Int}
    return (2, 0, 1)
end
function covstrparam(ct::SPPOWD_, ::Int)::Tuple{Int, Int}
    return (2, 1)
end

function covstrparam(ct::ZERO, ::Int)::Tuple{Int, Int}
    return (0, 0)
end
function covstrparam(ct::AbstractCovarianceType, ::Int)
    error("Unknown covariance type!")
end

################################################################################
# RCOEFNAMES
################################################################################
function rcoefnames(s, t, ct::SI_)
    return ["σ² "]
end
function rcoefnames(s, t, ct::SWC_)
    return ["σ² "]
end
function rcoefnames(s, t, ct::DIAG_)
    if isa(coefnames(s), AbstractArray{T,1} where T) l = length(coefnames(s)) else l = 1 end
    return fill!(Vector{String}(undef, l), "σ² ") .* string.(coefnames(s))
end
function rcoefnames(s, t, ct::Union{CS_, AR_})
    return ["σ² ", "ρ "]
end
function rcoefnames(s, t, ct::Union{CSH_, ARH_})
    cn = coefnames(s)
    if isa(cn, Vector)
        l  = length(cn)
    else
        l  = 1
    end
    v  = Vector{String}(undef, t)
    @. $(view(v, 1:l)) = "σ² " * string(cn)
    v[end] = "ρ "
    return v
end
function rcoefnames(s, t, ct::ARMA_)
    return ["σ² ", "γ ", "ρ "]
end
function rcoefnames(s, t, ct::Union{TOEP_, TOEPP_})
    v = Vector{String}(undef, t)
    v[1] = "σ² "
    if length(v) > 1
        for i = 2:length(v)
            v[i] = "ρ band $(i-1) "
        end
    end
    return v
end
function rcoefnames(s, t, ct::Union{TOEPH_, TOEPHP_})
    cn = coefnames(s)
    if isa(cn, Vector)
        l  = length(cn)
    else
        l  = 1
    end
    v  = Vector{String}(undef, t)
    @. $(view(v, 1:l)) = "σ² " * string(cn)
    if length(v) > l
        for i = l+1:length(v)
            v[i] = "ρ band $(i-l) "
        end
    end
    return v
end
function rcoefnames(s, t, ct::Union{SPEXP_, SPGAU_})
    return ["σ² ", "θ "]
end
function rcoefnames(s, t, ct::SPPOW_)
    return ["σ² ", "ρ "]
end
function rcoefnames(s, t, ct::Union{SPEXPD_, SPGAUD_})
    return ["σ² ", "σ²s ", "θ "]
end
function rcoefnames(s, t, ct::SPPOWD_)
    return ["σ² ", "σ²s ", "ρ "]
end

function indfromtn(ind, s)
    b = 0
    m = 0
    for i in 1:s-1
        b += s - i
        if b >= ind
            m = i
            break
        end
    end
    return m, s + ind - b
end

function rcoefnames(s, t, ct::UN_)
    cn = coefnames(s)
    if isa(cn, Vector)
        l  = length(cn)
    else
        l  = 1
    end
    v  = Vector{String}(undef, t)
    view(v, 1:l) .= string.(cn)
    if l > 1
        for i = 1:t-l
            m, n = indfromtn(i, l)
            v[i+l] = "ρ: " * v[m] * " × " * v[n]
        end
    end
    @. $(view(v, 1:l)) = "σ² " * $(view(v, 1:l))
    return v
end

function rcoefnames(s, t, ct::AbstractCovarianceType)
    v = Vector{String}(undef, t)
    v .= "Val "
    return v
end
################################################################################
# APPLY COV SCHEMA
################################################################################
function applycovschema!(::AbstractCovarianceType, ::Any)
    nothing
end

function applycovschema!(ct::SWC_{<:AbstractMatrix{T}}, vcovblock) where T
    if length(ct.wtsb) == 0
        for i in eachindex(vcovblock)
            push!(ct.wtsb, ct.wtsm[vcovblock[i], vcovblock[i]])
        end
    end
    return ct
end

################################################################################
# SHOW
################################################################################
function Base.show(io::IO, ct::CovarianceType)
    print(io, "Covariance Type: $(ct.s)")
end

function Base.show(io::IO, ct::AbstractCovarianceType)
    print(io, "$(typeof(ct).name.name)")
end
function Base.show(io::IO, ct::SI_)
    print(io, "SI")
end
function Base.show(io::IO, ct::SWC_)
    print(io, "SWC")
end
function Base.show(io::IO, ct::DIAG_)
    print(io, "DIAG")
end
function Base.show(io::IO, ct::AR_)
    print(io, "AR")
end
function Base.show(io::IO, ct::ARH_)
    print(io, "ARH")
end
function Base.show(io::IO, ct::CS_)
    print(io, "CS")
end
function Base.show(io::IO, ct::CSH_)
    print(io, "CSH")
end
function Base.show(io::IO, ct::ARMA_)
    print(io, "ARMA")
end
function Base.show(io::IO, ct::TOEP_)
    print(io, "TOEP")
end
function Base.show(io::IO, ct::TOEPP_)
    print(io, "TOEPP($(ct.p))")
end
function Base.show(io::IO, ct::TOEPH_)
    print(io, "TOEPH")
end
function Base.show(io::IO, ct::TOEPHP_)
    print(io, "TOEPHP($(ct.p))")
end
function Base.show(io::IO, ct::SPEXP_)
    print(io, "SPEXP")
end
function Base.show(io::IO, ct::SPPOW_)
    print(io, "SPPOW")
end
function Base.show(io::IO, ct::SPGAU_)
    print(io, "SPGAU")
end
function Base.show(io::IO, ct::SPEXPD_)
    print(io, "SPEXPD")
end
function Base.show(io::IO, ct::SPPOWD_)
    print(io, "SPPOWD")
end
function Base.show(io::IO, ct::SPGAUD_)
    print(io, "SPGAUD")
end
function Base.show(io::IO, ct::UN_)
    print(io, "UN")
end
function Base.show(io::IO, ct::ZERO)
    print(io, "No effect")
end
