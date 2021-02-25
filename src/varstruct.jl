################################################################################
#                     @covstr macro
################################################################################
"""
    @covstr(ex)

Macros for random/repeated effect model.

Example: @covstr(factor|subject)
"""
macro covstr(ex)
    return :(@formula(nothing ~ $ex).rhs)
end
function modelparse(term::FunctionTerm{typeof(|)})
    eff, subj = term.args_parsed
end

################################################################################
#                          COVARIANCE TYPE
################################################################################
struct CovarianceType{T} <: AbstractCovarianceType
    s::Symbol          #Covtype name
    p::T
    function CovarianceType(s, p)
        new{typeof(p)}(s, p)
    end
    function CovarianceType(s)
        CovarianceType(s, zero(Int))
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
    CovarianceType(:SI)
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
    CovarianceType(:DIAG)
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
    CovarianceType(:AR)
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
    CovarianceType(:ARH)
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
    CovarianceType(:CS)
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
    CovarianceType(:CSH)
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
    CovarianceType(:ARMA)
end
const ARMA = AutoregressiveMovingAverage()

"""
    Toeplitz()

Toeplitz covariance type. Only for G matrix.

TOEP = Toeplitz()

"""
function Toeplitz()
    CovarianceType(:TOEP)
end
const TOEP = Toeplitz()

"""
    ToeplitzParameterized(p::Int)

Toeplitz covariance type with parameter p, (number of bands = p - 1, if p = 1 its equal SI structure).

TOEPP(p) = ToeplitzParameterized(p)

"""
function ToeplitzParameterized(p::Int)
    CovarianceType(:TOEPP, p)
end
const TOEPP(p) = ToeplitzParameterized(p)

"""
    HeterogeneousToeplitz()

Heterogeneous toeplitz covariance type. Only for G matrix.

TOEPH = HeterogeneousToeplitz()

"""
function HeterogeneousToeplitz()
    CovarianceType(:TOEPH)
end
const TOEPH = HeterogeneousToeplitz()

"""
    HeterogeneousToeplitzParameterized(p::Int)

Heterogeneous toeplitz covariance type with parameter p, (number of bands = p - 1, if p = 1 its equal DIAG structure).

TOEPHP(p) = HeterogeneousToeplitzParameterized(p)

"""
function HeterogeneousToeplitzParameterized(p::Int)
    CovarianceType(:TOEPHP, p)
end
const TOEPHP(p) = HeterogeneousToeplitzParameterized(p)

#Spatial Power ?
#=
"""
    RZero()
"""
=#
function RZero()
    CovarianceType(:ZERO)
end

"""
    CustomCovarianceStruct(nparamf::Function, xmat!::Function)

* `nparamf` - function type (t, q) -> (a, b)
where:
t - size(z, 2) - number of levels for effect (number of columns of individual z matriz);
q - number of factors in the effect model;
a - number of variance parameters;
b - number of ρ parameters;

Example: (t, q) -> (t, 1) for CSH structure; (t, q) -> (1, 1) for AR, ets.

Tuple{Int, Int} should be returned.

* `xmat!` - construction function

G matrix function should update mx, where mx is zero matrix, p - parameter of CovarianceType structure, example:

```julia
function gmat_diag!(mx, θ::Vector{T}, p) where T
    for i = 1:size(mx, 1)
            mx[i, i] = θ[i] ^ 2
        end
    nothing
end
```
R matrix function should add R part to mx, rz - is repeated effect matrix, example:

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
struct CustomCovarianceStruct
    nparamf::Function
    xmat!::Function
end

"""
    CustomCovarianceType(ccs::CustomCovarianceStruct)

Make custom covariance type with CustomCovarianceStruct.
"""
CustomCovarianceType(ccs::CustomCovarianceStruct) = CovarianceType(:FUNC, ccs)

function covstrparam(ct::CovarianceType, t::Int, q::Int)::Tuple{Int, Int}
    if ct.s == :SI
        return (1, 0)
    elseif ct.s == :DIAG
        return (t, 0)
    elseif ct.s == :VC
        return (q, 0)
    elseif ct.s == :AR
        return (1, 1)
    elseif ct.s == :ARH
        return (t, 1)
    elseif ct.s == :ARMA
        return (1, 2)
    elseif ct.s == :CS
        return (1, 1)
    elseif ct.s == :CSH
        return (t, 1)
    elseif ct.s == :TOEP
        return (1, t - 1)
    elseif ct.s == :TOEPH
        return (t, t - 1)
    elseif ct.s == :TOEPP
        return (1, ct.p - 1)
    elseif ct.s == :TOEPHP
        return (t, ct.p - 1)
    elseif ct.s == :UN
        return (t, t * (t + 1) / 2 - t)
    elseif ct.s == :ZERO
        return (0, 0)
    elseif ct.s == :FUNC
        return ct.p.nparamf(t, q)
    else
        error("Unknown covariance type!")
    end
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

* `covtype` - covariance type (SI, DIAG, CS, CSH, AR, ARH, ARMA)
"""
struct VarEffect
    formula::FunctionTerm
    model::Union{Tuple{Vararg{AbstractTerm}}, AbstractTerm}
    covtype::CovarianceType
    coding::Dict{Symbol, AbstractContrasts}
    subj::AbstractTerm
    p::Int
    function VarEffect(formula, covtype::T, coding) where T <: AbstractCovarianceType
        model, subj = modelparse(formula)
        p = nterms(model)
        if coding === nothing
            coding = Dict{Symbol, AbstractContrasts}()
        end
        #if !isa(subj, Union{CategoricalTerm,ConstantTerm,InteractionTerm{<:NTuple{N,CategoricalTerm} where {N}},}) error("subject (blocking) variables must be Categorical") end
        new(formula, model, covtype, coding, subj, p)
    end
    function VarEffect(formula, covtype::T; coding = nothing) where T <: AbstractCovarianceType
        VarEffect(formula, covtype, coding)
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
    # subject (local) blocks for each effect
    block::Vector{Vector{Vector{UInt32}}}
    # blocks for vcov matrix / variance blocking factor (subject)
    vcovblock::Vector{Vector{UInt32}}
    # Z matrix
    z::Matrix{T}
    #subjz::Vector{BitArray{2}}
    # Blocks for each blocking subject, each effect, each effect subject
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
    #--
    function CovStructure(random, repeated, data)
        alleffl =  length(random) + 1
        #
        q       = Vector{Int}(undef, alleffl)
        t       = Vector{Int}(undef, alleffl)
        tr      = Vector{UnitRange{Int}}(undef, alleffl)
        schema  = Vector{Union{AbstractTerm, Tuple}}(undef, alleffl)
        block   = Vector{Vector{Vector{UInt32}}}(undef, alleffl)
        z       = Matrix{Float64}(undef, size(data, 1), 0)
        subjz   = Vector{BitMatrix}(undef, alleffl)
        zrndur  = Vector{UnitRange{Int}}(undef, alleffl - 1)
        rz      = Matrix{Float64}(undef, size(data, 1), 0)
        #Theta parameter type
        ct  = Vector{Symbol}(undef, 0)
        # Names
        rcnames = Vector{String}(undef, 0)
        #
        sn      = zeros(Int, alleffl)
        if length(random) > 1
            for i = 2:length(random)
                if random[i].covtype.s == :ZERO error("One of the random effect have zero type!") end
            end
        end
        # RANDOM EFFECTS
        for i = 1:length(random)
            if length(random[i].coding) == 0
                fill_coding_dict!(random[i].model, random[i].coding, data)
            end
            schema[i] = apply_schema(random[i].model, StatsModels.schema(data, random[i].coding))
            ztemp     = modelcols(MatrixTerm(schema[i]), data)
            q[i]      = size(ztemp, 2)
            csp       = covstrparam(random[i].covtype, q[i], random[i].p)
            t[i]      = csp[1] + csp[2]
            z         = hcat(z, ztemp)
            fillur!(zrndur, i, q)
            fillur!(tr, i, t)

            subjz[i]  = modelcols(MatrixTerm(apply_schema(random[i].subj, StatsModels.schema(data, fulldummycodingdict(random[i].subj)))), data)
            block[i]  = makeblocks(subjz[i])

            updatenametype!(ct, rcnames, csp, schema[i], random[i].covtype.s)
        end
        # REPEATED EFFECTS
        if length(repeated.coding) == 0
            fill_coding_dict!(repeated.model, repeated.coding, data)
        end

        schema[end] = apply_schema(repeated.model, StatsModels.schema(data, repeated.coding))
        rz          = modelcols(MatrixTerm(schema[end]), data)

        subjz[end]  = modelcols(MatrixTerm(apply_schema(repeated.subj, StatsModels.schema(data, fulldummycodingdict(repeated.subj)))), data)
        block[end]  = makeblocks(subjz[end])

        q[end]      = size(rz, 2)
        csp         = covstrparam(repeated.covtype, q[end], repeated.p)
        t[end]      = csp[1] + csp[2]
        tr[end]     = UnitRange(sum(t[1:end-1]) + 1, sum(t[1:end-1]) + t[end])
        updatenametype!(ct, rcnames, csp, schema[end], repeated.covtype.s)
        #Theta length
        tl  = sum(t)
        ########################################################################
        if random[1].covtype.s != :ZERO
            subjblockmat = subjz[1]
            if length(subjz) > 2
                for i = 2:length(subjz)-1
                    subjblockmat = noncrossmodelmatrix(subjblockmat, subjz[2])
                end
            end
            if !(repeated.covtype.s ∈ [:SI, :DIAG, :VC])
                subjblockmat = noncrossmodelmatrix(subjblockmat, subjz[end])
            end
        else
            subjblockmat = subjz[end]
        end
        blocks = makeblocks(subjblockmat)
        sblock = Vector{Vector{Vector{Vector{UInt32}}}}(undef, length(blocks))
        ########################################################################
        for i = 1:length(blocks)
            sblock[i] = Vector{Vector{Vector{UInt32}}}(undef, alleffl)
            for s = 1:alleffl
                sblock[i][s] = Vector{Vector{UInt32}}(undef, 0)
                for col in eachcol(view(subjz[s], blocks[i], :))
                    #if any(col) push!(sblock[i][s], sort!(findall(x->x==true, col))) end
                    if any(col) push!(sblock[i][s], findall(x->x==true, col)) end
                end
                sn[s] += length(sblock[i][s])
            end
        end
        #
        new{eltype(z)}(random, repeated, schema, rcnames, block, blocks, z, sblock, zrndur, rz, q, t, tr, tl, ct, sn)
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
    append!(rcnames, rcoefnames(schema, csp[1]+csp[2], s))
end
################################################################################
#=
function subjmatrix!(subj, data, subjz, i)
    if length(subj) > 0
        if length(subj) == 1
            subjterm = Term(subj[1])
        else
            subjterm = InteractionTerm(Tuple(Term.(subj)))
        end
        subjdict = Dict{Symbol, AbstractContrasts}()
        fill_coding_dict!(subjterm, subjdict, data)
        subjz[i]    = BitArray(modelcols(apply_schema(subjterm, StatsModels.schema(data, subjdict)), data))
    else
        subjz[i]    = trues(size(data, 1),1)
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
function noncrossmodelmatrix(mx, my)
    size(mx, 2) > size(my, 2) ?  (mat = mx' * my; a = mx) : (mat = my' * mx; a = my)
    #mat = mat * mat'
    @inbounds for n = 1:size(mat, 2)-1
        fr = findfirst(x->!iszero(x), view(mat, n, :))
        if !isnothing(fr)
            @inbounds for m = fr:size(mat, 1)
                if !iszero(mat[m, n])
                    fc = findfirst(x->!iszero(x), view(mat, m, n+1:size(mat, 2)))
                    if !isnothing(fc)
                        @inbounds for c = n+fc:size(mat, 2)
                            if !iszero(mat[m, c])
                                view(mat, :, n) .+= view(mat, :, c)
                                fill!(view(mat, :, c), 0)
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
    result = a * res
    result
end
#=
function noncrossmodelmatrix(mx::BitMatrix, my::BitMatrix)
    size(mx, 2) > size(my, 2) ?  (mat = mx' * my; a = mx) : (mat = my' * mx; a = my)
    mat = mat * mat'
    @inbounds for n = 1:size(mat, 2)-1
        @inbounds for m = 1:size(mat, 1)
            if !iszero(mat[m, n])
                @inbounds for c = n+1:size(mat, 2)
                    if mat[m, c] > zero(Int)
                        view(mat, :, n) .+= view(mat, :, c)
                        fill!(view(mat, :, c), zero(Int))
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
    res = replace(x -> iszero(x) ?  zero(Int) : one(Int), view(mat, :, cols))
    result = a * res
    result
end
=#
################################################################################
#                            CONTRAST CODING
################################################################################
function fill_coding_dict!(t::FunctionTerm{T}, d::Dict, data) where T
end
function fill_coding_dict!(t::T, d::Dict, data) where T <: ConstantTerm
end
function fill_coding_dict!(t::T, d::Dict, data) where T <: Type{InterceptTerm}
end
function fill_coding_dict!(t::T, d::Dict, data) where T <: Term
    if typeof(data[!, t.sym]) <: CategoricalArray || !(typeof(data[!, t.sym]) <: Vector{T} where T <: Real)
        d[t.sym] = StatsModels.FullDummyCoding()
    end
end
function fill_coding_dict!(t::T, d::Dict, data) where T <: CategoricalTerm
    if typeof(data[!, t.sym])  <: CategoricalArray
        d[t.sym] = StatsModels.FullDummyCoding()
    end
end
function fill_coding_dict!(t::T, d::Dict, data) where T <: InteractionTerm
    for i in t.terms
        if typeof(data[!, i.sym])  <: CategoricalArray || !(typeof(data[!, i.sym]) <: Vector{T} where T <: Real)
            d[i.sym] = StatsModels.FullDummyCoding()
        end
    end
end
function fill_coding_dict!(t::T, d::Dict, data) where T <: Tuple
    for i in t
        if isa(i, Term)
            if typeof(data[!, i.sym]) <: CategoricalArray || !(typeof(data[!, i.sym]) <: Vector{T} where T <: Real)
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
function fulldummycodingdict(t::ConstantTerm)
    d = Dict{Symbol, AbstractContrasts}()
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
