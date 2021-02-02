################################################################################
#                         @covstr macro
################################################################################
"""
    @covstr(ex)

Macros for random/repeated effect model.
"""
macro covstr(ex)
    return :(@formula(nothing ~ $ex).rhs)
end
################################################################################
#                          COVARIANCE TYPE
################################################################################
struct CovarianceType{T} <: AbstractCovarianceType
    s::Symbol          #Covtype name
    t::T
    function CovarianceType(s, t)
        new{typeof(t)}(s, t)
    end
    function CovarianceType(s)
        CovarianceType(s, 0)
    end
end
################################################################################
"""
    ScaledIdentity()

Scaled identity covariance type.

SI = ScaledIdentity()
"""
function ScaledIdentity()
    CovarianceType(:SI)
end
const SI = ScaledIdentity()
"""
    Diag()
"""
function Diag()
    CovarianceType(:DIAG)
end
const DIAG = Diag()
"""
    Autoregressive()
"""
function Autoregressive()
    CovarianceType(:AR)
end
const AR = Autoregressive()
"""
    HeterogeneousAutoregressive()
"""
function HeterogeneousAutoregressive()
    CovarianceType(:ARH)
end
const ARH = HeterogeneousAutoregressive()
"""
    CompoundSymmetry()
"""
function CompoundSymmetry()
    CovarianceType(:CS)
end
const CS = CompoundSymmetry()
"""
    HeterogeneousCompoundSymmetry()
"""
function HeterogeneousCompoundSymmetry()
    CovarianceType(:CSH)
end
const CSH = HeterogeneousCompoundSymmetry()
"""
"""
function AutoregressiveMovingAverage()
    CovarianceType(:ARMA)
end
const ARMA = AutoregressiveMovingAverage()
"""
    RZero()
"""
function RZero()
    CovarianceType(:ZERO)
end

#TOE

#TOEH

#UNST

#ARMA

function covstrparam(ct::CovarianceType, q::Int, p::Int)::Tuple{Int, Int, Int}
    if ct.s == :SI
        return (1, 0, 1)
    elseif ct.s == :DIAG
        return (q, 0, q)
    elseif ct.s == :VC
        return (p, 0, p)
    elseif ct.s == :AR
        return (1, 1, 2)
    elseif ct.s == :ARH
        return (q, 1, q + 1)
    elseif ct.s == :ARMA
        return (1, 2, 3)
    elseif ct.s == :CS
        return (1, 1, 2)
    elseif ct.s == :CSH
        return (q, 1, q + 1)
    elseif ct.s == :TOEP
        return (1, q - 1, q)
    elseif ct.s == :TOEPH
        return (q, q - 1, 2 * q - 1)
    elseif ct.s == :TOEPB
        return (1, ct.t - 1, ct.t)
    elseif ct.s == :TOEPHB
        return (q, ct.t - 1, q + ct.t - 1)
    elseif ct.s == :UN
        return (q, q * (q + 1) / 2 - q, q * (q + 1) / 2)
    elseif ct.s == :ZERO
        return (0, 0, 0)
    elseif ct.s == :FUNC
        error("Not implemented!")
    else
        error("Unknown covariance type!")
    end
end

################################################################################
#                  EFFECT
################################################################################
"""
    VarEffect(model, covtype::T, coding; fulldummy = true, subj = nothing) where T <: AbstractCovarianceType

Random/repeated effect.
"""
struct VarEffect
    model::Union{Tuple{Vararg{AbstractTerm}}, Nothing, AbstractTerm}
    covtype::CovarianceType
    coding::Dict{Symbol, AbstractContrasts}
    subj::Vector{Symbol}
    p::Int
    function VarEffect(model, covtype::T, coding; subj = nothing) where T <: AbstractCovarianceType
        if isa(subj, Nothing)
            subj = Vector{Symbol}(undef, 0)
        elseif isa(subj, Symbol)
            subj = [subj]
        elseif isa(subj,  AbstractVector{Symbol})
            #
        else
            throw(ArgumentError("subj type should be Symbol or Vector{tymbol}"))
        end
        p = nterms(model)
        #=
        if isa(model, Term)
            p = 1
        elseif isa(model, Tuple)
            p = length(model)
        else
            p = 0
        end
        =#
        if coding === nothing && model !== nothing
            coding = Dict{Symbol, AbstractContrasts}()
        elseif coding === nothing && model === nothing
            coding = Dict{Symbol, AbstractContrasts}()
        end
        new(model, covtype, coding, subj, p)
    end
    function VarEffect(model, covtype::T; coding = nothing, subj = nothing) where T <: AbstractCovarianceType
        VarEffect(model, covtype, coding;  subj = subj)
    end
    function VarEffect(model; coding = nothing)
        VarEffect(model, SI, coding)
    end
    function VarEffect(covtype::T; coding = nothing, subj = nothing) where T <: AbstractCovarianceType
        VarEffect(@covstr(1), covtype, coding; subj = subj)
    end
    function VarEffect()
        VarEffect(@covstr(1), SI, Dict{Symbol, AbstractContrasts}())
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
    tr::Vector{UnitRange{UInt32}}
    # θ Parameter count
    tl::Int
    # Parameter type :var / :rho
    ct::Vector{Symbol}
    #--
    function CovStructure(random, repeated, data, blocks)
        alleffl =  length(random) + 1
        #
        q       = Vector{Int}(undef, alleffl)
        t       = Vector{Int}(undef, alleffl)
        tr      = Vector{UnitRange{UInt32}}(undef, alleffl)
        schema  = Vector{Union{AbstractTerm, Tuple}}(undef, alleffl)
        block   = Vector{Vector{Vector{UInt32}}}(undef, alleffl)
        z       = Matrix{Float64}(undef, size(data, 1), 0)
        subjz   = Vector{BitArray{2}}(undef, alleffl)
        sblock  = Vector{Vector{Vector{Vector{UInt32}}}}(undef, length(blocks))
        zrndur  = Vector{UnitRange{Int}}(undef, alleffl - 1)
        rz      = Matrix{Float64}(undef, size(data, 1), 0)
        #Theta parameter type
        ct  = Vector{Symbol}(undef, 0)
        # Names
        rcnames = Vector{String}(undef, 0)
        #
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
            if i > 1
                if  random[i].subj == random[i - 1].subj block[i] = block[i - 1] else block[i]  = intersectdf(data, random[i].subj) end
            else
                block[i]  = intersectdf(data, random[i].subj)
            end
            schema[i] = apply_schema(random[i].model, StatsModels.schema(data, random[i].coding))
            ztemp     = modelcols(MatrixTerm(schema[i]), data)
            q[i]      = size(ztemp, 2)
            csp       = covstrparam(random[i].covtype, q[i], random[i].p)
            t[i]      = csp[3]
            z         = hcat(z, ztemp)
            fillur!(zrndur, i, q)
            fillur!(tr, i, t)
            subjmatrix!(random[i].subj, data, subjz, i)
            updatenametype!(ct, rcnames, csp, schema[i], random[i].covtype.s)
        end
        # REPEATED EFFECTS
        if length(repeated.coding) == 0
            fill_coding_dict!(repeated.model, repeated.coding, data)
        end
        block[end]  = intersectdf(data, repeated.subj)
        schema[end] = apply_schema(repeated.model, StatsModels.schema(data, repeated.coding))
        rz          = modelcols(MatrixTerm(schema[end]), data)
        subjmatrix!(repeated.subj, data, subjz, length(subjz))
        q[end]      = size(rz, 2)
        csp         = covstrparam(repeated.covtype, q[end], repeated.p)
        t[end]      = csp[3]
        tr[end]     = UnitRange(sum(t[1:end-1]) + 1, sum(t[1:end-1]) + t[end])
        updatenametype!(ct, rcnames, csp, schema[end], repeated.covtype.s)
        #Theta length
        tl  = sum(t)
        ########################################################################
        ########################################################################
        for i = 1:length(blocks)
            sblock[i] = Vector{Vector{Vector{UInt32}}}(undef, alleffl)
            for s = 1:alleffl
                sblock[i][s] = Vector{Vector{UInt32}}(undef, 0)
                for col in eachcol(view(subjz[s], blocks[i], :))
                    if any(col) push!(sblock[i][s], sort!(findall(x->x==true, col))) end
                end
            end
        end
        #
        new{eltype(z)}(random, repeated, schema, rcnames, block, z, sblock, zrndur, rz, q, t, tr, tl, ct)
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
    append!(rcnames, rcoefnames(schema, csp[3], s))
end
################################################################################
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
################################################################################
#                            CONTRAST CODING
################################################################################

function fill_coding_dict!(t::T, d::Dict, data) where T <: ConstantTerm
end
function fill_coding_dict!(t::T, d::Dict, data) where T <: Type{InterceptTerm}
end
function fill_coding_dict!(t::T, d::Dict, data) where T <: Term
    if typeof(data[!, t.sym]) <: CategoricalArray
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
        if typeof(data[!, i.sym])  <: CategoricalArray
            d[i.sym] = StatsModels.FullDummyCoding()
        end
    end
end
function fill_coding_dict!(t::T, d::Dict, data) where T <: Tuple
    for i in t
        if isa(i, Term)
            if typeof(data[!, i.sym]) <: CategoricalArray
                d[i.sym] = StatsModels.FullDummyCoding()
            end
        else
            fill_coding_dict!(i, d, data)
        end
    end
end
################################################################################

################################################################################
function Base.show(io::IO, e::VarEffect)
    println(io, "Effect")
    println(io, "Model:", e.model)
    println(io, "Type: ", e.covtype)
    println(io, "Coding: ", e.coding)
    println(io, "Subject:", e.subj)
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
    println(io, "Covariance Type: $(ct.s)")
end
