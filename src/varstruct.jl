macro covstr(ex)
    return :(@formula(nothing ~ $ex).rhs)
end


struct VarianceComponents <: AbstractCovarianceType
    f
    v
    rho
    function VarianceComponents()
        new(x -> x, x -> x, x -> 0)
    end
end
VC = VarianceComponents()

struct ScaledIdentity <: AbstractCovarianceType
    f
    v
    rho
    function ScaledIdentity()
        new(x -> 1, x -> 1, x -> 0)
    end
end
SI = ScaledIdentity()

struct HeterogeneousCompoundSymmetry <: AbstractCovarianceType
    f
    v
    rho
    function HeterogeneousCompoundSymmetry()
        new(x -> x + 1, x -> x, x -> 1)
    end
end
CSH = HeterogeneousCompoundSymmetry()

#schema(df6)
#rschema = apply_schema(Term(formulation), schema(df, Dict(formulation => StatsModels.FullDummyCoding())))
#apply_schema(Term(formulation), shema1)
#Z   = modelcols(rschema, df)
#reduce(hcat, Z)

struct VarEffect{T <: AbstractCovarianceType}
    model
    covtype::T
    coding::Dict{Symbol, AbstractContrasts}
    function VarEffect(model, covtype::T, coding) where T <: AbstractCovarianceType
        if coding === nothing && model !== nothing
            coding = Dict{Symbol, AbstractContrasts}()
            fill_coding_dict(model, coding)
        end
        new{T}(model, covtype, coding)
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
    function VarEffect(model, covtype::T; coding = nothing) where T <: AbstractCovarianceType
        VarEffect(model, covtype, coding)
    end
end

struct CovStructure <: AbstractCovarianceStructure
    random::Vector{VarEffect}
    repeated::VarEffect
    z::Matrix                                         #Z matrix
    rz::Union{Matrix, Nothing}
    q::Vector{Int}
    t::Vector{Int}
    tr::Vector{UnitRange}
    tl::Int
    ct::Vector{Symbol}
    function CovStructure(random, repeated, data)
        q       = Vector{Int}(undef, length(random) + 1)
        t       = Vector{Int}(undef, length(random) + 1)
        tr      = Vector{UnitRange}(undef, length(random) + 1)

        rschema = apply_schema(random[1].model, schema(data, random[1].coding))
        if schemalength(rschema) == 1 z = modelcols(rschema, data) else z = reduce(hcat, modelcols(rschema, data)) end
        q[1]    = size(z, 2)
        t[1]    = random[1].covtype.f(q[1])
        tr[1]   = UnitRange(1, t[1])
        if length(random) > 1
            for i = 2:length(random)
                rschema = apply_schema(random[i].model, schema(data, random[i].coding))
                if schemalength(rschema) == 1 ztemp = modelcols(rschema, data) else ztemp = reduce(hcat, modelcols(rschema, data)) end
                q[i]    = size(ztemp, 2)
                t[i]    = random[i].covtype.f(q[i])
                z       = hcat(z, ztemp)
                tr[i]   = UnitRange(sum(t[1:i-1]) + 1, sum(t[1:i-1])+t[i])
            end
        end
        if repeated.model !== nothing
            rschema = apply_schema(repeated.model, schema(data, repeated.coding))
            if schemalength(rschema) == 1 rz = modelcols(rschema, data) else rz = reduce(hcat, modelcols(rschema, data)) end
            q[end]     = size(rz, 2)
        else
            rz         = nothing
            q[end]     = 0
        end
        t[end]      = repeated.covtype.f(q[end])
        tr[end]     = UnitRange(sum(t[1:end-1]) + 1, sum(t[1:end-1]) + t[end])
        tl  = sum(t)
        ct  = Vector{Symbol}(undef, tl)
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
        new(random, repeated, z, rz, q, t, tr, tl, ct)
    end
end

@inline function schemalength(s)
    if isa(s, Tuple)
        return length(s)
    else
        return 1
    end
end

@inline function gmat(θ::Vector{T}, zn, ve::VarEffect{VarianceComponents})::AbstractMatrix{T} where T
    Diagonal(θ .^ 2)
end

@inline function gmat(θ::Vector{T}, zn, ve::VarEffect{ScaledIdentity})::AbstractMatrix{T} where T
    I(zn)*(θ[1] ^ 2)
end

@inline function gmat(θ::Vector{T}, zn, ve::VarEffect{HeterogeneousCompoundSymmetry})::AbstractMatrix{T} where T
    mx = Matrix{T}(undef, zn, zn)
    for m = 1:zn
        @inbounds mx[m, m] = θ[m]
    end
    if zn > 1
        for m = 1:zn - 1
            for n = m + 1:zn
                @inbounds mx[m, n] = mx[m, m] * mx[n, n] * θ[end]
            end
        end
    end
    for m = 1:zn
        @inbounds mx[m, m] = mx[m, m] * mx[m, m]
    end
    Symmetric(mx)
end

@inline function gmat_blockdiag(θ::Vector{T}, covstr) where T
    vm = Vector{Matrix{T}}(undef, length(covstr.random))
    for i = 1:length(covstr.random)
        vm[i] = gmat(θ[covstr.tr[i]], covstr.q[i], covstr.random[i])
    end
    BlockDiagonal(vm)
end

@inline function rmat(θ::Vector{T}, rz, ve::VarEffect{VarianceComponents}) where T
    Diagonal(rz* (θ .^ 2))
end
@inline function rmat(θ::Vector{T}, rzn, ve::VarEffect{ScaledIdentity})::AbstractMatrix{T} where T
    I(rzn)*(θ[1] ^ 2)
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

function get_z_matrix(data, covstr::CovStructure)
    rschema = apply_schema(covstr.random.model, schema(data, covstr.random.coding))
    Z       = modelcols(rschema, data)
end

function gmat(covstr::CovStructure)

end

function get_term_vec(covstr::CovStructure)
    covstr.random.model
end
=#

@inline function fill_coding_dict(t::T, d::Dict) where T <: AbstractTerm
    d[t.sym] = StatsModels.FullDummyCoding()
end
@inline function fill_coding_dict(t::T, d::Dict) where T <: Tuple
    for i in t
        d[i.sym] = StatsModels.FullDummyCoding()
    end
end
#-------------------------------------------------------------------------------


#=
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
=#

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
