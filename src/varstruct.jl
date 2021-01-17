################################################################################
#                         @covstr macro
################################################################################
macro covstr(ex)
    return :(@formula(nothing ~ $ex).rhs)
end
################################################################################
#                       SIMPLE FUNCTIONS
################################################################################
function ffx(x::T)::T where T
    x
end
function ffxzero(x::T)::T where T
    zero(T)
end
function ffxone(x::T)::T where T
    one(T)
end
function ffxpone(x::T)::T where T
    x + one(T)
end
#=
function ffxmone(x::T)::T where T
    x - one(T)
end
function ff2xmone(x::T)::T where T
    2x - one(T)
end
=#
################################################################################
#                          COVARIANCE TYPE
################################################################################
struct CovarianceType <: AbstractCovarianceType
    s::Symbol          #Covtype name
    f::Function        #number of parameters for Z size 2
    v::Function        #number of variance parameters for Z size 2
    rho::Function      #number of rho parameters for Z size 2
end
################################################################################
function ScaledIdentity()
    CovarianceType(:SI, ffxone, ffxone, ffxzero)
end
const SI = ScaledIdentity()

function Diag()
    CovarianceType(:DIAG, ffx, ffx, ffxzero)
end
const DIAG = Diag()

function Autoregressive()
    CovarianceType(:AR, x -> 2, ffxone, ffxone)
end
const AR = Autoregressive()

function HeterogeneousAutoregressive()
    CovarianceType(:ARH, ffxpone, ffx, ffxone)
end
const ARH = HeterogeneousAutoregressive()

function CompoundSymmetry()
    CovarianceType(:CS, x -> 2, ffxone, ffxone)
end
const CS = CompoundSymmetry()

function HeterogeneousCompoundSymmetry()
    CovarianceType(:CSH, ffxpone, ffx, ffxone)
end
const CSH = HeterogeneousCompoundSymmetry()

function RZero()
    CovarianceType(:ZERO, x -> 0, x -> 0, x -> 0)
end

#TOE

#TOEH

#UNST

#ARMA

################################################################################
#                  EFFECT
################################################################################
"""
    VarEffect(model, covtype::T, coding; fulldummy = true, subj = nothing) where T <: AbstractCovarianceType
"""
struct VarEffect
    model::Union{Tuple{Vararg{AbstractTerm}}, Nothing, AbstractTerm}
    covtype::CovarianceType
    coding::Dict{Symbol, AbstractContrasts}
    fulldummy::Bool
    subj::Vector{Symbol}
    function VarEffect(model, covtype::T, coding; fulldummy = true, subj = nothing) where T <: AbstractCovarianceType
        #if isa(model, AbstractTerm) model = tuple(model) end
        if isa(subj, Nothing)
            subj = Vector{Symbol}(undef, 0)
        elseif isa(subj, Symbol)
            subj = [subj]
        elseif isa(subj,  AbstractVector{Symbol})
            #
        else
            throw(ArgumentError("subj type should be Symbol or Vector{tymbol}"))
        end

        if coding === nothing && model !== nothing
            coding = Dict{Symbol, AbstractContrasts}()
        elseif coding === nothing && model === nothing
            coding = Dict{Symbol, AbstractContrasts}()
        end
        #if isa(model, AbstractTerm) model = tuple(model) end
        new(model, covtype, coding, fulldummy, subj)
    end
    function VarEffect(model, covtype::T; coding = nothing, fulldummy = true, subj = nothing) where T <: AbstractCovarianceType
        VarEffect(model, covtype, coding; fulldummy = fulldummy, subj = subj)
    end


    function VarEffect(model; coding = nothing)
        VarEffect(model, DIAG, coding)
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
    # number of parametert in each effect
    t::Vector{Int}
    # range of each parameters in θ vector
    tr::Vector{UnitRange{UInt32}}
    # θ Parameter count
    tl::UInt16
    # Parameter type :var / :rho
    ct::Vector{Symbol}
    #--
    #
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
        #
        # RANDOM EFFECTS
        for i = 1:length(random)
            if length(random[i].coding) == 0 && random[i].fulldummy
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
            t[i]      = random[i].covtype.f(q[i])
            z         = hcat(z, ztemp)
            fillur!(zrndur, i, q)
            fillur!(tr, i, t)
            subjmatrix!(random[i].subj, data, subjz, i)
        end
        # REPEATED EFFECTS
        if length(repeated.coding) == 0 && repeated.fulldummy
            fill_coding_dict!(repeated.model, repeated.coding, data)
        end
        block[end]  = intersectdf(data, repeated.subj)
        schema[end] = apply_schema(repeated.model, StatsModels.schema(data, repeated.coding))
        rz          = modelcols(MatrixTerm(schema[end]), data)
        subjmatrix!(repeated.subj, data, subjz, length(subjz))
        q[end]      = size(rz, 2)
        t[end]      = repeated.covtype.f(q[end])
        tr[end]     = UnitRange(sum(t[1:end-1]) + 1, sum(t[1:end-1]) + t[end])
        #Theta length
        tl  = sum(t)
        #Theta parameter type
        ct  = Vector{Symbol}(undef, tl)
        # Names
        rcnames = Vector{String}(undef, tl)
        ctn = 1
        for i = 1:length(random)
            if random[i].covtype.s == :ZERO
                continue
            end
            for i2 = 1:random[i].covtype.v(q[i])
                ct[ctn] = :var
                ctn +=1
            end
            if random[i].covtype.rho(q[i]) > 0
                for i2 = 1:random[i].covtype.rho(q[i])
                    ct[ctn] = :rho
                    ctn +=1
                end
            end
            view(rcnames, tr[i]) .= rcoefnames(schema[i], t[i], Val{random[i].covtype.s}())
        end
        for i2 = 1:repeated.covtype.v(q[end])
            ct[ctn] = :var
            ctn +=1
        end
        if repeated.covtype.rho(q[end]) > 0
            for i2 = 1:repeated.covtype.rho(q[end])
                ct[ctn] = :rho
                ctn +=1
            end
        end
        view(rcnames, tr[end]) .= rcoefnames(schema[end], t[end], Val{repeated.covtype.s}())
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
#=
function schemalength(s)
    if isa(s, Tuple)
        return length(s)
    else
        return 1
    end
end
=#
#

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
"""
    Return variance-covariance matrix V
"""
#=
function vmat()::AbstractMatrix

end
=#
################################################################################
function Base.show(io::IO, e::VarEffect)
    println(io, "Effect")
    println(io, "Model:", e.model)
    println(io, "Type: ", e.covtype)
    println(io, "Coding: ", e.coding)
    println(io, "FullDummy: ", e.fulldummy)
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
