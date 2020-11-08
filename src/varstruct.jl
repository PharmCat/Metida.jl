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
function ffxmone(x::T)::T where T
    x - one(T)
end
function ff2xmone(x::T)::T where T
    2x - one(T)
end

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

function VarianceComponents()
    CovarianceType(:VC, ffx, ffx, ffxzero)
end
const VC = VarianceComponents()

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
    model::Union{Tuple{Vararg{AbstractTerm}}, Nothing}
    covtype::CovarianceType
    coding::Dict{Symbol, AbstractContrasts}
    fulldummy::Bool
    subj::Union{Symbol, Nothing}
    function VarEffect(model, covtype::T, coding; fulldummy = true, subj = nothing) where T <: AbstractCovarianceType
        if coding === nothing && model !== nothing
            coding = Dict{Symbol, AbstractContrasts}()
        elseif coding === nothing && model === nothing
            coding = Dict{Symbol, AbstractContrasts}()
        end
        if isa(model, AbstractTerm) model = tuple(model) end
        new(model, covtype, coding, fulldummy, subj)
    end
    function VarEffect(model, covtype::T; coding = nothing, fulldummy = true, subj = nothing) where T <: AbstractCovarianceType
        VarEffect(model, covtype, coding; fulldummy = fulldummy, subj = subj)
    end


    function VarEffect(model; coding = nothing)
        VarEffect(model, VC, coding)
    end
    function VarEffect(covtype::T; coding = nothing) where T <: AbstractCovarianceType
        VarEffect(nothing, covtype, coding)
    end
    function VarEffect()
        VarEffect(nothing, SI, Dict{Symbol, AbstractContrasts}())
    end
end
################################################################################
#                            COVARIANCE STRUCTURE
################################################################################
struct CovStructure{T} <: AbstractCovarianceStructure
    random::Vector{VarEffect}                                                   #Random effects
    repeated::VarEffect                                                         #Repearted effects
    schema::Vector{Tuple}
    rcnames::Vector{String}
    block::Vector{Vector{Vector{Int}}}
    z::Matrix{T}                                                                #Z matrix
    zr::Vector{UnitRange{Int}}
    rz::Matrix{T}                                                               #repeated effect parametrization matrix
    q::Vector{Int}                                                              # size 2 of z/rz matrix
    t::Vector{Int}                                                              # number of parametert in each effect
    tr::Vector{UnitRange{Int}}                                                  # range of each parameters in θ vector
    tl::Int                                                                     # θ Parameter count
    ct::Vector{Symbol}                                                          #Parameter type :var / :rho
    function CovStructure(random, repeated, data)
        q       = Vector{Int}(undef, length(random) + 1)
        t       = Vector{Int}(undef, length(random) + 1)
        tr      = Vector{UnitRange}(undef, length(random) + 1)
        schema  = Vector{Tuple}(undef, length(random) + 1)
        block   = Vector{Vector{Vector{Int}}}(undef, length(random))
        z       = Matrix{Float64}(undef, size(data, 1), 0)
        zr      = Vector{UnitRange}(undef, length(random))
            for i = 1:length(random)
                if length(random[i].coding) == 0 && random[i].fulldummy
                    fill_coding_dict!(random[i].model, random[i].coding, data)
                end
                if i > 1
                    if  random[i].subj == random[i - 1].subj block[i] = block[i - 1] else block[i]  = subjblocks(data, random[i].subj) end
                else
                    block[i]  = subjblocks(data, random[i].subj)
                end
                schema[i] = apply_schema(random[i].model, StatsModels.schema(data, random[i].coding))
                ztemp   = reduce(hcat, modelcols(schema[i], data))
                #schema[i] = rschema
                q[i]    = size(ztemp, 2)
                t[i]    = random[i].covtype.f(q[i])
                z       = hcat(z, ztemp)
                fillur!(zr, i, q)
                fillur!(tr, i, t)
                #=
                if i > 1
                    zr[i]   = UnitRange(sum(q[1:i-1]) + 1, sum(q[1:i-1]) + q[i])
                else
                    zr[1]   = UnitRange(1, q[1])
                end
                if i > 1
                    tr[i]   = UnitRange(sum(t[1:i-1]) + 1, sum(t[1:i-1])+t[i])
                else
                    tr[1]   = UnitRange(1, t[1])
                end
                =#
            end
        if repeated.model !== nothing
            if length(repeated.coding) == 0 && repeated.fulldummy
                fill_coding_dict!(repeated.model, repeated.coding, data)
            end
            schema[end] = apply_schema(repeated.model, StatsModels.schema(data, repeated.coding))
            rz = reduce(hcat, modelcols(schema[end], data))
            #schema[end] = rschema
            q[end]     = size(rz, 2)
        else
            rz           = Matrix{eltype(z)}(undef, 0, 0)
            schema[end]  = tuple(0)
            q[end]       = 0
        end
        t[end]      = repeated.covtype.f(q[end])
        tr[end]     = UnitRange(sum(t[1:end-1]) + 1, sum(t[1:end-1]) + t[end])
        tl  = sum(t)
        ct  = Vector{Symbol}(undef, tl)
        rcnames = Vector{String}(undef, tl)
        ctn = 1
        for i = 1:length(random)
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
        new{eltype(z)}(random, repeated, schema, rcnames, block, z, zr, rz, q, t, tr, tl, ct)
    end
end
################################################################################
function fillur!(ur, i, v)
    if i > 1
        ur[i]   = UnitRange(sum(v[1:i-1]) + 1, sum(v[1:i-1]) + v[i])
    else
        ur[1]   = UnitRange(1, v[1])
    end
end
################################################################################
function schemalength(s)
    if isa(s, Tuple)
        return length(s)
    else
        return 1
    end
end
#
################################################################################
#                            CONTRAST CODING
################################################################################

ContinuousTerm

function fill_coding_dict!(t::T, d::Dict, data) where T <: ConstantTerm
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
function vmat(G, R, Z)::AbstractMatrix
    return  mulαβαtc(Z, G, R)
end
################################################################################
function vmatvec(G, Z, θ)
    v = Vector{Matrix{eltype(G)}}(undef, length(Z))
    for i = 1:length(Z)
        v[i] = mulαβαtc(Z[i], G, rmat(θ[1:2], Z[i]))
    end
    #reduce(vcat, v)
    v
end

################################################################################
