macro covstr(ex)
    return :(@formula(nothing ~ $ex).rhs)
end


struct VarianceComponents <: AbstractCovarianceType
    f
    function VarianceComponents()
        new(x -> x)
    end
end
VC = VarianceComponents

struct ScaledIdentity <: AbstractCovarianceType
    f
    function ScaledIdentity()
        new(x -> 1)
    end
end
SI = ScaledIdentity

struct HeterogeneousCompoundSymmetry <: AbstractCovarianceType
    f
    function HeterogeneousCompoundSymmetry()
        new(x -> x + 1)
    end
end
CSH = HeterogeneousCompoundSymmetry

#schema(df6)
#rschema = apply_schema(Term(formulation), schema(df, Dict(formulation => StatsModels.FullDummyCoding())))
#apply_schema(Term(formulation), shema1)
#Z   = modelcols(rschema, df)
#reduce(hcat, Z)

struct VarEffect
    model
    covtype::AbstractCovarianceType
    coding::Dict{Symbol, AbstractContrasts}
    function VarEffect(model, covtype, coding)
        if coding === nothing && model !== nothing
            coding = Dict{Symbol, AbstractContrasts}()
            fill_coding_dict(model, coding)
        end
        new(model, covtype, coding)
    end
    function VarEffect(model; coding = nothing)
        VarEffect(model, VarianceComponents(), coding)
    end
    function VarEffect()
        VarEffect(nothing, ScaledIdentity(), Dict{Symbol, AbstractContrasts}())
    end
    function VarEffect(model, covtype::Type; coding = nothing)
        VarEffect(model, covtype(), coding)
    end
end

struct CovStructure{T <: Union{VarEffect, Vector{VarEffect}}} <: AbstractCovarianceStructure
    random::T
    repeated::VarEffect
    z
    q
    t
    function CovStructure(random::Vector{VarEffect}, repeated, data)
        q       = Vector{Int}(undef, length(random))
        t       = Vector{Int}(undef, length(random) + 1)
        rschema = apply_schema(random[1].model, schema(data, random[1].coding))
        z       = modelcols(rschema, data)
        q[1]    = size(z, 2)
        t[1]    = random[1].covtype.f(q[1])
        if length(random) > 1
            for i = 1:length(random)
                rschema = apply_schema(random[i].model, schema(data, random[i].coding))
                ztemp   = modelcols(rschema, data)
                q[i]    = size(ztemp, 2)
                t[i]    = random[i].covtype.f(q[i])
                z       = hcat(z, ztemp)
            end
        end
        if repeated.model !== nothing
            rschema = apply_schema(repeated.model, schema(data, repeated.coding))
            rz      = modelcols(rschema, data)
            rzn     = size(rz, 2)
        else
            rz      = nothing
            rzn     = 0
        end
        t[end]      = repeated.covtype.f(rzn)
        new{typeof(random)}(random, repeated, z, q, t)
    end
end


#=
function get_z_matrix(data, covstr::CovStructure{Vector{VarEffect}})
    rschema = apply_schema(covstr.random[1].model, schema(data, covstr.random[1].coding))
    Z       = modelcols(rschema, data)
    if length(covstr.random) > 1
        for i = 1:length(covstr.random)
            rschema = apply_schema(covstr.random[i].model, schema(data, covstr.random[i].coding))
            Z       = hcat(modelcols(rschema, data))
        end
    end
    Z
end
=#
function get_z_matrix(data, covstr::CovStructure{VarEffect})
    rschema = apply_schema(covstr.random.model, schema(data, covstr.random.coding))
    Z       = modelcols(rschema, data)
end

function gmat(covstr::CovStructure)

end

function get_term_vec(covstr::CovStructure{VarEffect})
    covstr.random.model
end

function fill_coding_dict(t::T, d::Dict) where T <: AbstractTerm
    d[t.sym] = StatsModels.FullDummyCoding()
end
function fill_coding_dict(t::T, d::Dict) where T <: Tuple
    for i in t
        d[i.sym] = StatsModels.FullDummyCoding()
    end
end
#-------------------------------------------------------------------------------



"""
    G matrix
"""
@inline function gmat(σ::AbstractVector)::AbstractMatrix
    cov = sqrt(σ[1] * σ[2]) * σ[3]
    return Symmetric([σ[1] cov; cov σ[2]])
end


"""
    R matrix (ForwardDiff+)
"""
@inline function rmat(σ::AbstractVector, Z::AbstractMatrix)::Matrix
    return Diagonal(Z*σ)
end
"""
    Return variance-covariance matrix V
"""
@inline function vmat(G, R, Z)::AbstractMatrix
    return  mulαβαtc(Z, G, R)
end

################################################################################

@inline function vmatvec(G, Z, θ)
    v = Vector{Matrix{eltype(G)}}(undef, length(Z))
    for i = 1:length(Z)
        v[i] = mulαβαtc(Z[i], G, rmat(θ[1:2], Z[i]))
    end
    #reduce(vcat, v)
    v
end
