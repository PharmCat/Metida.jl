macro covstr(ex)
    return :(@formula(nothing ~ $ex).rhs)
end


struct VarianceComponents <: AbstractCovarianceType
end
VC = VarianceComponents

struct ScaledIdentity <: AbstractCovarianceType
end
SI = ScaledIdentity

struct HeterogeneousCompoundSymmetry <: AbstractCovarianceType
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
    coding
    function VarEffect(model, covtype::AbstractCovarianceType; coding = nothing)
        new(model, covtype, coding)
    end
    function VarEffect(model; coding = nothing)
        new(model, VarianceComponents(), coding)
    end
    function VarEffect()
        new(nothing, ScaledIdentity(), nothing)
    end
    function VarEffect(model, covtype::Type; coding = nothing)
        new(model, covtype(), coding)
    end
end

struct CovStructure <: AbstractCovarianceStructure
    random
    repeated::VarEffect
end

function get_term_vec(covstr::CovStructure)
    covstr.random.model
end

function filltdict(t::T, d::Dict) where T <: AbstractTerm
    d[t.sym] = StatsModels.FullDummyCoding()
end
function filltdict(t::T, d::Dict) where T <: Tuple
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
