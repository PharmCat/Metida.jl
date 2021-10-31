################################################################################
#                     @covstr macro
################################################################################
"""
    @covstr(ex)

Macros for random/repeated effect model.

# Example

```julia
@covstr(factor|subject)
```
"""
macro covstr(ex)
    return :(@formula(nothing ~ $ex).rhs)
end
function modelparse(term::FunctionTerm{typeof(|)})
    eff, subj = term.args_parsed
end

################################################################################
# COVMAT METHOD
################################################################################
#=
"""
    CovmatMethod(nparamf, xmat!)

* `nparamf` - function type (t, q) -> (a, b, c)
where:
`t` - size(z, 2) - number of levels for effect (number of columns of individual z matriz);
`q` - number of factors in the effect model;
`a` - number of variance parameters;
`b` - number of ρ parameters;
`c` - other parameers (optional).

Example: `(t, q) -> (t, 1)` for CSH structure; `(t, q) -> (1, 1)` for AR, ets.

`Tuple{Int, Int}` should be returned.

* `xmat!` - construction function

G matrix function should update `mx`, where `mx` is zero matrix, `p` - parameter of CovarianceType structure.

# Exapple

```julia
function gmat_diag!(mx, θ::Vector{T}, p) where T
    for i = 1:size(mx, 1)
            mx[i, i] = θ[i] ^ 2
        end
    nothing
end
```

# Exapple

R matrix function should add R part to `mx`, `rz` - is repeated effect matrix.

# Example

```julia
function rmatp_diag!(mx, θ::Vector{T}, rz, p) where T
    for i = 1:size(mx, 1)
        for c = 1:length(θ)
            mx[i, i] += rz[i, c] * θ[c] * θ[c]
        end
    end
    nothing
end
```
"""
struct CovmatMethod{Nf, Xf} <: AbstractCovmatMethod
    nparamf::Nf
    xmat!::Xf
    function CovmatMethod(nparamf, xmat!)
        new{typeof(nparamf), typeof(xmat!)}(nparamf, xmat!)
    end
    function CovmatMethod()
        CovmatMethod(nothing, nothing)
    end
end
=#
################################################################################
#abstract type AbstractCType end
struct SI_ <: AbstractCovarianceType end
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
struct UN_ <: AbstractCovarianceType end
struct ZERO <: AbstractCovarianceType end


################################################################################
#                          COVARIANCE TYPE
################################################################################
"""
    CovarianceType(cm::AbstractCovmatMethod)

Make covariance type with CovmatMethod.

# Example

```julia
customg = CovarianceType(CovmatMethod((q,p) -> (q, 1), Metida.gmat_csh!))
```
"""
struct CovarianceType
    s::AbstractCovarianceType
    z::Bool
    function CovarianceType(s::AbstractCovarianceType, z::Bool)
        new(s, z)
    end
    function CovarianceType(s::AbstractCovarianceType)
        CovarianceType(s,  true)
    end
end



#=
struct FullCovarianceType <: AbstractCovarianceType
    g::CovarianceType
    r::CovarianceType
end
=#
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

!!! warning
    Experimental

Spatian Exponential covariance structure. Used only for repeated effect.

```math
R_{i,j} = \\sigma^{2} * exp(-dist(i,j)/exp(\\theta))
```

where `dist` - Euclidean distance between row-vectors of repeated effect matrix for subject `i` and `j`.

SPEXP = SpatialExponential()

"""
function SpatialExponential() #Spatial Power ?
    CovarianceType(SPEXP_())
end
const SPEXP = SpatialExponential()


function RZero()
    CovarianceType(ZERO(), false)
end

function covstrparam(ct::SI_, ::Int)::Tuple{Int, Int}
    return (1, 0)
end
function covstrparam(ct::DIAG_, t::Int, )::Tuple{Int, Int}
    return (t, 0)
end
function covstrparam(ct::Union{AR_, CS_}, ::Int)::Tuple{Int, Int}
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
    return (1, ct.p - 1)
end
function covstrparam(ct::TOEPHP_, t::Int)::Tuple{Int, Int}
    return (t, ct.p - 1)
end
function covstrparam(ct::UN_, t::Int)::Tuple{Int, Int}
    return (t, t * (t + 1) / 2 - t)
end
function covstrparam(ct::SPEXP_, ::Int)::Tuple{Int, Int, Int}
    return (1, 0, 1)
end
function covstrparam(ct::ZERO, t::Int)::Tuple{Int, Int}
    return (t, t * (t + 1) / 2 - t)
end
function covstrparam(ct::AbstractCovarianceType, ::Int)
    error("Unknown covariance type!")
end

################################################################################
#                  EFFECT
################################################################################
"""
    VarEffect(formula, covtype::T, coding) where T <: AbstractCovarianceType

    VarEffect(formula, covtype::T; coding = nothing) where T <: AbstractCovarianceType

    VarEffect(formula; coding = nothing)

Random/repeated effect.

* `formula` from @covstr(ex) macros.

* `covtype` - covariance type (SI, DIAG, CS, CSH, AR, ARH, ARMA, TOEP, TOEPH, TOEPP, TOEPHP)

!!! note

    Categorical factors are coded with `FullDummyCoding()` by default, use `coding` for other contrast codeing.

# Example

```julia
VarEffect(@covstr(1+factor|subject), CSH)

VarEffect(@covstr(1 + formulation|subject), CSH; coding = Dict(:formulation => StatsModels.DummyCoding()))
```
"""
struct VarEffect
    formula::FunctionTerm
    model::Union{Tuple{Vararg{AbstractTerm}}, AbstractTerm}
    covtype::CovarianceType
    coding::Dict{Symbol, AbstractContrasts}
    subj::Union{Tuple{Vararg{AbstractTerm}}, AbstractTerm}
    p::Int
    function VarEffect(formula, covtype::CovarianceType, coding)
        model, subj = modelparse(formula)
        p = nterms(model)
        if coding === nothing
            coding = Dict{Symbol, AbstractContrasts}()
        end
        #if !isa(subj, Union{CategoricalTerm,ConstantTerm,InteractionTerm{<:NTuple{N,CategoricalTerm} where {N}},}) error("subject (blocking) variables must be Categorical") end
        new(formula, model, covtype, coding, subj, p)
    end
    function VarEffect(formula, covtype::CovarianceType; coding = nothing)
        VarEffect(formula, covtype, coding)
    end
    function VarEffect(formula, covtype::AbstractCovarianceType; coding = nothing)
        VarEffect(formula, CovarianceType(covtype), coding)
    end
    function VarEffect(formula; coding = nothing)
        VarEffect(formula, SI, coding)
    end
end
################################################################################
#                            COVARIANCE STRUCTURE
################################################################################
struct CovStructure{T} <: AbstractCovarianceStructure
    # Random effects
    random::Vector{VarEffect}
    # Repearted effects
    repeated::VarEffect
    schema::Vector{Union{Tuple, AbstractTerm}}
    rcnames::Vector{String}
    # blocks for vcov matrix / variance blocking factor (subject)
    vcovblock::Vector{Vector{UInt32}}
    # number of random effect
    rn::Int
    # Z matrix
    z::Matrix{T}
    #subjz::Vector{BitArray{2}}
    # Blocks for each blocking subject, each effect, each effect subject sblock[block][rand eff][subj]
    sblock::Vector{Vector{Vector{Vector{UInt32}}}}
    #unit range z column range for each random effect
    zrndur::Vector{UnitRange{Int}}
    # repeated effect parametrization matrix
    rz::Matrix{T}
    # size 2 of z/rz matrix
    q::Vector{Int}
    # total number of parameters in each effect
    t::Vector{Int}
    # range of each parameters in θ vector
    tr::Vector{UnitRange{Int}}
    # θ Parameter count
    tl::Int
    # Parameter type :var / :rho
    ct::Vector{Symbol}
    # Nubber of subjects in each effect
    sn::Vector{Int}
    # Maximum number per block
    maxn::Int
    #--
    function CovStructure(random, repeated, data)
        alleffl =  length(random) + 1
        rown    =  length(Tables.rows(data))
        #
        q       = Vector{Int}(undef, alleffl)
        t       = Vector{Int}(undef, alleffl)
        tr      = Vector{UnitRange{Int}}(undef, alleffl)
        schema  = Vector{Union{AbstractTerm, Tuple}}(undef, alleffl)
        z       = Matrix{Float64}(undef, rown, 0)
        subjz   = Vector{BitMatrix}(undef, alleffl)
        zrndur  = Vector{UnitRange{Int}}(undef, alleffl - 1)
        rz      = Matrix{Float64}(undef, rown, 0)
        rn      = length(random)
        #Theta parameter type
        ct  = Vector{Symbol}(undef, 0)
        # Names
        rcnames = Vector{String}(undef, 0)
        #
        sn      = zeros(Int, alleffl)
        if rn > 1
            @inbounds for i = 2:rn
                if !random[i].covtype.z error("One of the random effect have zero type!") end
            end
        end
        # RANDOM EFFECTS
        @inbounds for i = 1:rn
            if length(random[i].coding) == 0
                fill_coding_dict!(random[i].model, random[i].coding, data)
            end
            schema[i] = apply_schema(random[i].model, StatsModels.schema(data, random[i].coding))
            ztemp     = modelcols(MatrixTerm(schema[i]), data)
            q[i]      = size(ztemp, 2)
            csp       = covstrparam(random[i].covtype.s, q[i])
            t[i]      = sum(csp)
            z         = hcat(z, ztemp)
            fillur!(zrndur, i, q)
            fillur!(tr, i, t)

            subjz[i]  = convert(BitMatrix, modelcols(MatrixTerm(apply_schema(random[i].subj, StatsModels.schema(data, fulldummycodingdict(random[i].subj)))), data))
            sn[i] = size(subjz[i], 2)
            updatenametype!(ct, rcnames, csp, schema[i], random[i].covtype.s)
        end
        # REPEATED EFFECTS
        if length(repeated.coding) == 0
            fill_coding_dict!(repeated.model, repeated.coding, data)
        end

        schema[end] = apply_schema(repeated.model, StatsModels.schema(data, repeated.coding))
        rz          = modelcols(MatrixTerm(schema[end]), data)

        subjz[end]  = convert(BitMatrix, modelcols(MatrixTerm(apply_schema(repeated.subj, StatsModels.schema(data, fulldummycodingdict(repeated.subj)))), data))
        sn[end] = size(subjz[end], 2)
        q[end]      = size(rz, 2)
        csp         = covstrparam(repeated.covtype.s, q[end])
        t[end]      = sum(csp)
        tr[end]     = UnitRange(sum(t[1:end-1]) + 1, sum(t[1:end-1]) + t[end])
        updatenametype!(ct, rcnames, csp, schema[end], repeated.covtype.s)
        #Theta length
        tl  = sum(t)
        ########################################################################
        if random[1].covtype.z
            subjblockmat = subjz[1]
            if length(subjz) > 2
                for i = 2:length(subjz)-1
                    subjblockmat = noncrossmodelmatrix(subjblockmat, subjz[2])
                end
            end
            if !(isa(repeated.covtype.s, SI_) || isa(repeated.covtype.s, DIAG_))
                subjblockmat = noncrossmodelmatrix(subjblockmat, subjz[end])
            end
        else
            subjblockmat = subjz[end]
        end
        blocks = makeblocks(subjblockmat) #vcovblock
        sblock = Vector{Vector{Vector{Vector{UInt32}}}}(undef, length(blocks))
        ########################################################################
        @inbounds for i = 1:length(blocks)
            sblock[i] = Vector{Vector{Vector{UInt32}}}(undef, alleffl)
            @inbounds for s = 1:alleffl
                sblock[i][s] = Vector{Vector{UInt32}}(undef, 0)
                @inbounds for col in eachcol(view(subjz[s], blocks[i], :))
                    if any(col) push!(sblock[i][s], findall(col)) end
                end
            end
        end
        #
        maxn = 0
        for i in blocks
            lvcb = length(i)
            if lvcb > maxn maxn = lvcb end
        end
        new{eltype(z)}(random, repeated, schema, rcnames, blocks, rn, z, sblock, zrndur, rz, q, t, tr, tl, ct, sn, maxn)
    end
end
################################################################################
function fillur!(ur, i, v)
    if i > 1
        ur[i]   = UnitRange(sum(v[1:i-1]) + 1, sum(v[1:i-1]) + v[i])
    else
        if v[1] > 0
            ur[1]   = UnitRange(1, v[1])
        else
            ur[1]   = UnitRange(0, 0)
        end
    end
end
################################################################################
function updatenametype!(ct, rcnames, csp, schema, s)
    append!(ct, fill!(Vector{Symbol}(undef, csp[1]), :var))
    append!(ct, fill!(Vector{Symbol}(undef, csp[2]), :rho))
    if length(csp) == 3 append!(ct, fill!(Vector{Symbol}(undef, csp[3]), :theta)) end
    append!(rcnames, rcoefnames(schema, sum(csp), s))
end
################################################################################
function rcoefnames(s, t, ct::SI_)
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
    view(v, 1:l) .= (fill!(Vector{String}(undef, l), "σ² ") .*string.(cn))
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
    view(v, 1:l) .= (fill!(Vector{String}(undef, l), "σ² ") .*string.(cn))
    if length(v) > l
        for i = l+1:length(v)
            v[i] = "ρ band $(i-l) "
        end
    end
    return v
end
function rcoefnames(s, t, ct::SPEXP_)
    return ["σ² ", "θ "]
end
function rcoefnames(s, t, ct::AbstractCovarianceType)
    v = Vector{String}(undef, t)
    v .= "Val"
    return v
end
#=
function rcoefnames(s, t, ve)
    if ve == SI
        return ["σ² "]
    elseif ve == DIAG
        if isa(coefnames(s), AbstractArray{T,1} where T) l = length(coefnames(s)) else l = 1 end
        return fill!(Vector{String}(undef, l), "σ² ") .* string.(coefnames(s))
    elseif ve == CS || ve == AR
        return ["σ² ", "ρ "]
    elseif ve == CSH || ve == ARH
        cn = coefnames(s)
        if isa(cn, Vector)
            l  = length(cn)
        else
            l  = 1
        end
        v  = Vector{String}(undef, t)
        view(v, 1:l) .= (fill!(Vector{String}(undef, l), "σ² ") .*string.(cn))
        v[end] = "ρ "
        return v
    elseif ve == ARMA
        return ["σ² ", "γ ", "ρ "]
    elseif ve == TOEP || ve == TOEPP
        v = Vector{String}(undef, t)
        v[1] = "σ² "
        if length(v) > 1
            for i = 2:length(v)
                v[i] = "ρ band $(i-1) "
            end
        end
        return v
    elseif ve == TOEPH || ve == TOEPHP
        cn = coefnames(s)
        if isa(cn, Vector)
            l  = length(cn)
        else
            l  = 1
        end
        v  = Vector{String}(undef, t)
        view(v, 1:l) .= (fill!(Vector{String}(undef, l), "σ² ") .*string.(cn))
        if length(v) > l
            for i = l+1:length(v)
                v[i] = "ρ band $(i-l) "
            end
        end
        return v
    else
        v = Vector{String}(undef, t)
        v .= "NA"
        return v
    end
end
=#
################################################################################
function makeblocks(subjz)
    blocks = Vector{Vector{UInt32}}(undef, 0)
    for i = 1:size(subjz, 2)
        b = findall(x->!iszero(x), view(subjz, :, i))
        if length(b) > 0 push!(blocks, b) end
    end
    blocks
end
################################################################################
function noncrossmodelmatrix(mx::AbstractArray, my::AbstractArray)
    size(mx, 2) > size(my, 2) ?  (mat = mx' * my; a = mx) : (mat = my' * mx; a = my)
    #mat = mat * mat'
    T = eltype(mat)
    @inbounds for n = 1:size(mat, 2)-1
        fr = findfirst(x->!iszero(x), view(mat, n, :))
        if !isnothing(fr)
            @inbounds for m = fr:size(mat, 1)
                if !iszero(mat[m, n])
                    fc = findfirst(x->!iszero(x), view(mat, m, n+1:size(mat, 2)))
                    if !isnothing(fc)
                        @inbounds for c = n+fc:size(mat, 2)
                            if !iszero(mat[m, c])
                                #view(mat, :, n) .+= view(mat, :, c)
                                A = view(mat, :, n)
                                broadcast!(+, A, A, view(mat, :, c))
                                fill!(view(mat, :, c), zero(T))
                            end
                        end
                    end
                end
            end
        end
    end
    cols = Vector{Int}(undef, 0)
    @inbounds for i = 1:size(mat, 2)
        if !iszero(sum(view(mat,:, i)))
            push!(cols, i)
        end
    end
    res = replace(x -> iszero(x) ?  0 : 1, view(mat, :, cols))
    a * res
end
################################################################################
#                            CONTRAST CODING
################################################################################

function fill_coding_dict!(t::T, d::Dict, data) where T <: Union{ConstantTerm, InterceptTerm, FunctionTerm}
end
function fill_coding_dict!(t::T, d::Dict, data) where T <: Term
    if typeof(Tables.getcolumn(data, t.sym)) <: CategoricalArray || !(typeof(Tables.getcolumn(data, t.sym)) <: Vector{T} where T <: Real)
        d[t.sym] = StatsModels.FullDummyCoding()
    end
end
#function fill_coding_dict!(t::T, d::Dict, data) where T <: CategoricalTerm
#    if typeof(data[!, t.sym])  <: CategoricalArray
#        d[t.sym] = StatsModels.FullDummyCoding()
#    end
#end
function fill_coding_dict!(t::T, d::Dict, data) where T <: InteractionTerm
    for i in t.terms
        if typeof(Tables.getcolumn(data, i.sym))  <: CategoricalArray || !(typeof(Tables.getcolumn(data, i.sym)) <: Vector{T} where T <: Real)
            d[i.sym] = StatsModels.FullDummyCoding()
        end
    end
end
function fill_coding_dict!(t::T, d::Dict, data) where T <: Tuple{Vararg{AbstractTerm}}
    for i in t
        if isa(i, Term)
            if typeof(Tables.getcolumn(data, i.sym)) <: CategoricalArray || !(typeof(Tables.getcolumn(data, i.sym)) <: Vector{T} where T <: Real)
                d[i.sym] = StatsModels.FullDummyCoding()
            end
        else
            fill_coding_dict!(i, d, data)
        end
    end
end
function fulldummycodingdict(t::InteractionTerm)
    d = Dict{Symbol, AbstractContrasts}()
    for i in t.terms
        d[i.sym] = StatsModels.FullDummyCoding()
    end
    d
end
function fulldummycodingdict(t::T) where T <: Union{CategoricalTerm, Term}
    d = Dict{Symbol, AbstractContrasts}()
    d[t.sym] = StatsModels.FullDummyCoding()
    d
end
function fulldummycodingdict(t::T) where T <: Union{ConstantTerm, InterceptTerm}
    d = Dict{Symbol, AbstractContrasts}()
    d
end
function fulldummycodingdict(t::Tuple{Vararg{AbstractTerm}})
    d = Dict{Symbol, AbstractContrasts}()
    for i in t
        merge!(d, fulldummycodingdict(i))
    end
    d
end

################################################################################

################################################################################
function Base.show(io::IO, e::VarEffect)
    println(io, "  Formula: ", e.formula)
    println(io, "  Effect model: ", e.model)
    println(io, "  Subject model: ", e.subj)
    println(io, "  Type: ", e.covtype.s)
    print(io, "  User coding:")
    if length(e.coding) > 0
        for (k, v) in e.coding
            print(io, " $(k) => $(v);")
        end
    else
        print(io, " No")
    end
end

function Base.show(io::IO, cs::CovStructure)
    println(io, "Covariance Structure:")
    for i = 1:length(cs.random)
        println(io, "Random $(i):", cs.random[i])
    end
    println(io, "Repeated: ", cs.repeated)
    println(io, "Random effect range in complex Z: ", cs.zrndur)
    println(io, "Size of Z: ", cs.q)
    println(io, "Parameter number for each effect: ", cs.t)
    println(io, "Theta length:", cs.tl)
end

function Base.show(io::IO, ct::CovarianceType)
    print(io, "Covariance Type: $(ct.s)")
end

function Base.show(io::IO, ct::AbstractCovarianceType)
    print(io, "$(typeof(ct).name.name)")
end
function Base.show(io::IO, ct::SI_)
    print(io, "SI")
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
function Base.show(io::IO, ct::UN_)
    print(io, "UN")
end
function Base.show(io::IO, ct::ZERO)
    print(io, "No effect")
end
