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
    subj::Vector{Symbol}
    function VarEffect(model, covtype::T, coding; fulldummy = true, subj = nothing) where T <: AbstractCovarianceType
        if isa(model, AbstractTerm) model = tuple(model) end
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
        VarEffect(@covstr(1), covtype, coding)
    end
    function VarEffect()
        VarEffect(@covstr(1), SI, Dict{Symbol, AbstractContrasts}())
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
    subjz
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
        block   = Vector{Vector{Vector{Int}}}(undef, length(random) + 1 )
        z       = Matrix{Float64}(undef, size(data, 1), 0)
        subjz   = Vector{BitArray{2}}(undef, length(random) + 1)
        zr      = Vector{UnitRange}(undef, length(random))
        rz      = Matrix{Float64}(undef, size(data, 1), 0)
            for i = 1:length(random)
                if length(random[i].coding) == 0 && random[i].fulldummy
                    fill_coding_dict!(random[i].model, random[i].coding, data)
                end
                if i > 1
                    #if  random[i].subj == random[i - 1].subj block[i] = block[i - 1] else block[i]  = subjblocks(data, random[i].subj) end
                    if  random[i].subj == random[i - 1].subj block[i] = block[i - 1] else block[i]  = intersectdf(data, random[i].subj) end
                else
                    #block[i]  = subjblocks(data, random[i].subj)
                    block[i]  = intersectdf(data, random[i].subj)
                end

                schema[i] = apply_schema(random[i].model, StatsModels.schema(data, random[i].coding))
                #ztemp     = reduce(hcat, modelcols(schema[i], data)) #MatrixTerm should be used modelcols(MatrixTerm(schema[i]), data)
                ztemp     = modelcols(MatrixTerm(schema[i]), data)
                #schema[i] = rschema
                q[i]      = size(ztemp, 2)
                t[i]      = random[i].covtype.f(q[i])
                z         = hcat(z, ztemp)
                fillur!(zr, i, q)
                fillur!(tr, i, t)
                if length(random[i].subj) > 0
                    sujterm = InteractionTerm(Tuple(term.(random[i].subj)))
                    subjdict = Dict{Symbol, AbstractContrasts}()
                    fill_coding_dict!(sujterm, subjdict, data)
                    subjz[i]    = BitArray(modelcols(apply_schema(sujterm, StatsModels.schema(data, subjdict)), data))
                else
                    subjz[i]    = trues(size(data, 1),1)
                end


            end
        #if repeated.model !== nothing
            if length(repeated.coding) == 0 && repeated.fulldummy
                fill_coding_dict!(repeated.model, repeated.coding, data)
            end
            block[end]  = intersectdf(data, repeated.subj)
            schema[end] = apply_schema(repeated.model, StatsModels.schema(data, repeated.coding))
            rz          = hcat(rz, reduce(hcat, modelcols(schema[end], data)))

            if length(repeated.subj) > 0
                sujterm = InteractionTerm(Tuple(term.(repeated.subj)))
                subjdict = Dict{Symbol, AbstractContrasts}()
                fill_coding_dict!(sujterm, subjdict, data)
                subjz[end]    = BitArray(modelcols(apply_schema(sujterm, StatsModels.schema(data, subjdict)), data))
            else
                subjz[end]    = trues(size(data, 1),1)
            end

            #schema[end] = rschema
            q[end]      = size(rz, 2)
        #else
        #    rz           = Matrix{eltype(z)}(undef, 0, 0)
        #    schema[end]  = tuple(0)
        #    q[end]       = 0
        #end
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
        new{eltype(z)}(random, repeated, schema, rcnames, block, z, subjz, zr, rz, q, t, tr, tl, ct)
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
