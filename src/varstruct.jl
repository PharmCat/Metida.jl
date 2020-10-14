macro covstr(ex)
    return :(@formula(nothing ~ $ex).rhs)
end

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

struct CovarianceType <: AbstractCovarianceType
    s::Symbol
    f::Function
    v::Function
    rho::Function
end

function VarianceComponents()
    CovarianceType(:VC, ffx, ffx, ffxzero)
end
const VC = CovarianceType(:VC, ffx, ffx, ffxzero)

function ScaledIdentity()
    CovarianceType(:SI, ffxone, ffxone, ffxzero)
end

const SI = CovarianceType(:SI, ffxone, ffxone, ffxzero)

function HeterogeneousCompoundSymmetry()
    CovarianceType(:CSH, ffxpone, ffx, ffxone)
end
const CSH = CovarianceType(:CSH, ffxpone, ffx, ffxone)


function Autoregressive()
    CovarianceType(:AR, x -> 2, ffxone, ffxone)
end
const AR = CovarianceType(:AR, x -> 2, ffxone, ffxone)


function HeterogeneousAutoregressive()
    CovarianceType(:ARH, ffxpone, ffx, ffxone)
end
const ARH = CovarianceType(:ARH, ffxpone, ffx, ffxone)

#=
struct Toepiz <: AbstractCovarianceType
    f::Function
    v::Function
    rho::Function
    function Toepiz()
        new(ffx, ffxone, ffxmone)
    end
end
TOEP = Toepiz()

struct HeterogeneousToepiz <: AbstractCovarianceType
    f::Function
    v::Function
    rho::Function
    function HeterogeneousToepiz()
        new(ff2xmone, ffx, ffxmone)
    end
end
TOEPH = HeterogeneousToepiz()

struct BandToepiz <: AbstractCovarianceType
    f::Function
    v::Function
    rho::Function
    n::Int
    function BandToepiz(n)
        new(ffx, ffxone, ffxmone, n)
    end
end
TOEPB = BandToepiz(1)
=#

struct VarEffect
    model::Union{Tuple{Vararg{AbstractTerm}}, Nothing}
    covtype::CovarianceType
    coding::Dict{Symbol, AbstractContrasts}
    function VarEffect(model, covtype::T, coding) where T <: AbstractCovarianceType
        if coding === nothing && model !== nothing
            coding = Dict{Symbol, AbstractContrasts}()
            fill_coding_dict(model, coding)
        elseif coding === nothing && model === nothing
            coding = Dict{Symbol, AbstractContrasts}()
        end
        if isa(model, AbstractTerm) model = tuple(model) end
        new(model, covtype, coding)
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
    function VarEffect(model, covtype::T; coding = nothing) where T <: AbstractCovarianceType
        VarEffect(model, covtype, coding)
    end
end

struct CovStructure{T} <: AbstractCovarianceStructure
    random::Vector{VarEffect}
    repeated::VarEffect
    ves::Vector{Symbol}
    schema::Vector{Tuple}
    rcnames::Vector{String}
    z::Matrix{T}                                                                   #Z matrix
    rz::Matrix{T}
    q::Vector{Int}
    t::Vector{Int}
    tr::Vector{UnitRange{Int}}
    tl::Int                                                                     #Parameter count
    ct::Vector{Symbol}                                                          #Parameter type :var / :rho
    function CovStructure(random, repeated, data)
        ves     = Vector{Symbol}(undef, length(random) + 1)
        q       = Vector{Int}(undef, length(random) + 1)
        t       = Vector{Int}(undef, length(random) + 1)
        tr      = Vector{UnitRange}(undef, length(random) + 1)
        schema  = Vector{Tuple}(undef, length(random) + 1)
        rschema = apply_schema(random[1].model, StatsModels.schema(data, random[1].coding))
        #if schemalength(rschema) == 1 z = modelcols(rschema, data) else z = reduce(hcat, modelcols(rschema, data)) end
        z       = reduce(hcat, modelcols(rschema, data))
        schema[1] = rschema
        ves[1]  = random[1].covtype.s
        q[1]    = size(z, 2)
        t[1]    = random[1].covtype.f(q[1])
        tr[1]   = UnitRange(1, t[1])
        if length(random) > 1
            for i = 2:length(random)
                rschema = apply_schema(random[i].model, StatsModels.schema(data, random[i].coding))
                #if schemalength(rschema) == 1 ztemp = modelcols(rschema, data) else ztemp = reduce(hcat, modelcols(rschema, data)) end
                ztemp = reduce(hcat, modelcols(rschema, data))
                schema[i] = rschema
                ves[i]  = random[i].covtype.s
                q[i]    = size(ztemp, 2)
                t[i]    = random[i].covtype.f(q[i])
                z       = hcat(z, ztemp)
                tr[i]   = UnitRange(sum(t[1:i-1]) + 1, sum(t[1:i-1])+t[i])
            end
        end
        if repeated.model !== nothing
            rschema = apply_schema(repeated.model, StatsModels.schema(data, repeated.coding))
            #if schemalength(rschema) == 1 rz = modelcols(rschema, data) else rz = reduce(hcat, modelcols(rschema, data)) end
            rz = reduce(hcat, modelcols(rschema, data))
            schema[end] = rschema
            q[end]     = size(rz, 2)
        else
            rz           = Matrix{eltype(z)}(undef, 0, 0)
            schema[end]  = tuple(0)
            q[end]       = 0
        end
        ves[end]    = repeated.covtype.s
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

        new{eltype(z)}(random, repeated, ves, schema, rcnames, z, rz, q, t, tr, tl, ct)
    end
end

@inline function schemalength(s)
    if isa(s, Tuple)
        return length(s)
    else
        return 1
    end
end

function gmat(θ::Vector{T}, zn::Int, ::CovarianceType, ::Val{:SI}) where T
    Matrix{T}(I(zn)*(θ[1] ^ 2))
    #I(zn)*(θ[1] ^ 2)
end
function gmat_si!(mx, θ::Vector{T}, zn::Int, ::CovarianceType) where T
    val = θ[1] ^ 2
    for i = 1:size(mx, 1)
        mx[i, i] = val
    end
    nothing
end

function gmat(θ::Vector{T}, ::Int, ::CovarianceType, ::Val{:VC}) where T
    Matrix{T}(Diagonal(θ .^ 2))
    #Diagonal(θ .^ 2)
end
function gmat_vc!(mx, θ::Vector{T}, ::Int, ::CovarianceType) where T
    for i = 1:size(mx, 1)
        mx[i, i] = θ[i]
    end
    nothing
end

function gmat(θ::Vector{T}, zn::Int, ::CovarianceType, ::Val{:AR}) where T
    mx  = Matrix{T}(undef, zn, zn)
    mx .= θ[1] ^ 2
    if zn > 1
        for m = 1:zn - 1
            for n = m + 1:zn
                @inbounds mx[m, n] = mx[m, m] * θ[2] ^ (n - m)
            end
        end
    end
    Matrix{T}(Symmetric(mx))
    #Symmetric(mx)
end
function gmat_ar!(mx, θ::Vector{T}, zn::Int, ::CovarianceType) where T
    mx .= θ[1] ^ 2
    if zn > 1
        for m = 1:zn - 1
            for n = m + 1:zn
                @inbounds mx[m, n] = mx[m, m] * θ[2] ^ (n - m)
                @inbounds mx[n, m] = mx[m, n]
            end
        end
    end
    nothing
end


function gmat(θ::Vector{T}, zn::Int, ::CovarianceType, ::Val{:ARH}) where T
    mx  = Matrix{T}(undef, zn, zn)
    for m = 1:zn
        @inbounds mx[m, m] = θ[m]
    end
    if zn > 1
        for m = 1:zn - 1
            for n = m + 1:zn
                @inbounds mx[m, n] = mx[m, m] * mx[n, n] * θ[end] ^ (n - m)
            end
        end
    end
    for m = 1:zn
        @inbounds mx[m, m] = mx[m, m] * mx[m, m]
    end
    Matrix{T}(Symmetric(mx))
    #Symmetric(mx)
end
function gmat_arh!(mx, θ::Vector{T}, ::Int, ::CovarianceType) where T
    for m = 1:size(mx, 1)
        @inbounds mx[m, m] = θ[m]
    end
    if zn > 1
        for m = 1:zn - 1
            for n = m + 1:zn
                @inbounds mx[m, n] = mx[m, m] * mx[n, n] * θ[end] ^ (n - m)
                @inbounds mx[n, m] = mx[m, n]
            end
        end
    end
    for m = 1:zn
        @inbounds mx[m, m] = mx[m, m] * mx[m, m]
    end
    nothing
end


function gmat(θ::Vector{T}, zn::Int, ::CovarianceType, ::Val{:CSH}) where T
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
    Matrix{T}(Symmetric(mx))
    #Symmetric(mx)
end
function gmat_csh!(mx, θ::Vector{T}, ::Int, ::CovarianceType) where T
    s = size(mx, 1)
    for m = 1:s
        @inbounds mx[m, m] = θ[m]
    end
    if s > 1
        for m = 1:s - 1
            for n = m + 1:s
                @inbounds mx[m, n] = mx[m, m] * mx[n, n] * θ[end]
                @inbounds mx[n, m] = mx[m, n]
            end
        end
    end
    for m = 1:s
        @inbounds mx[m, m] = mx[m, m] * mx[m, m]
    end
    nothing
end
#=
function gmat_blockdiag(θ::Vector{T}, covstr) where T
    vm = Vector{AbstractMatrix{T}}(undef, length(covstr.ves) - 1)
    for i = 1:length(covstr.random)
        vm[i] = gmat(θ[covstr.tr[i]], covstr.q[i], covstr.random[i].covtype, Val{covstr.random[i].covtype.s}()) #covstr.random[i])
    end
    BlockDiagonal(vm)
end
=#
function gmat_base(θ::Vector{T}, covstr) where T
    q = size(covstr.z, 2)
    mx = zeros(T, q, q)
    for i = 1:length(covstr.random)
        s = 1 + sum(covstr.q[1:i]) - covstr.q[i]
        e = sum(covstr.q[1:i])
        #mx[s:e, s:e] .= gmat(θ[covstr.tr[i]], covstr.q[i], covstr.random[i].covtype, Val{covstr.random[i].covtype.s}())
        if covstr.random[i].covtype.s == :SI
            gmat_si!(view(mx,s:e, s:e), θ[covstr.tr[i]], covstr.q[i], covstr.random[i].covtype)
        elseif covstr.random[i].covtype.s == :VC
            gmat_vc!(view(mx,s:e, s:e), θ[covstr.tr[i]], covstr.q[i], covstr.random[i].covtype)
        elseif covstr.random[i].covtype.s == :AR
            gmat_ar!(view(mx,s:e, s:e), θ[covstr.tr[i]], covstr.q[i], covstr.random[i].covtype)
        elseif covstr.random[i].covtype.s == :ARH
            gmat_arh!(view(mx,s:e, s:e), θ[covstr.tr[i]], covstr.q[i], covstr.random[i].covtype)
        elseif covstr.random[i].covtype.s == :CSH
            gmat_csh!(view(mx,s:e, s:e), θ[covstr.tr[i]], covstr.q[i], covstr.random[i].covtype)
        else
            throw(ErrorException("Unknown covariance structure: $(covstr.random[i].covtype.s), n = $(i)"))
        end
        #gmat!(view(mx,s:e, s:e), θ[covstr.tr[i]], covstr.q[i], covstr.random[i].covtype, Val{covstr.random[i].covtype.s}())
    end
    mx
end

################################################################################

function rmat_basep!(mx, θ::AbstractVector{T}, covstr) where T

end

@inline function rmat(θ::Vector{T}, rz, rn, ct, ::Val{:SI}) where T
    I(rn) * (θ[1] ^ 2)
end
@inline function rmatp_si!(mx, θ::Vector{T}, ::Matrix, ::Int, ::CovarianceType) where T
    θsq = θ[1]*θ[1]
    for i = 1:size(mx, 1)
            mx[i, i] += θsq
    end
    nothing
end

@inline function rmat(θ::Vector{T}, rz, rn, ct, ::Val{:VC}) where T
    Diagonal(rz * (θ .^ 2))
end
@inline function rmatp_vc!(mx, θ::Vector{T}, rz, ::Int, ::CovarianceType) where T
    for i = 1:size(mx, 1)
        for c = 1:length(θ)
            mx[i, i] += θ[c]*θ[c]*rz[i, c]
        end
    end
    nothing
end

@inline function rmat(θ::Vector{T}, rz, rn, ct, ::Val{:AR}) where T
    mx  = Matrix{T}(undef, rn, rn)
    mx .= θ[1] ^ 2
    if rn > 1
        for m = 1:rn - 1
            for n = m + 1:rn
                @inbounds mx[m, n] = mx[m, m] * θ[2] ^ (n - m)
            end
        end
    end
    Symmetric(mx)
end
@inline function rmat(θ::Vector{T}, rz, rn, ct, ::Val{:ARH}) where T
    mx   = Matrix(Diagonal(rz * (θ[1:end-1])))
    if rn > 1
        for m = 1:rn - 1
            for n = m + 1:rn
                @inbounds mx[m, n] = mx[m, m] * mx[n, n] * θ[end] ^ (n - m)
            end
        end
    end
    for m = 1:rn
        @inbounds mx[m, m] = mx[m, m] * mx[m, m]
    end
    Symmetric(mx)
end
@inline function rmat(θ::Vector{T}, rz, rn,  ct, ::Val{:CSH}) where T #???
    mx   = Matrix(Diagonal(rz * (θ[1:end-1])))
    if rn > 1
        for m = 1:rn - 1
            for n = m + 1:rn
                @inbounds mx[m, n] = mx[m, m] * mx[n, n] * θ[end]
            end
        end
    end
    for m = 1:rn
        @inbounds mx[m, m] = mx[m, m] * mx[m, m]
    end
    Symmetric(mx)
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

function get_term_vec(covstr::CovStructure)
    covstr.random.model
end
=#
@inline function fill_coding_dict(t::T, d::Dict) where T <: ConstantTerm
end
@inline function fill_coding_dict(t::T, d::Dict) where T <: Term
    d[t.sym] = StatsModels.FullDummyCoding()
end
@inline function fill_coding_dict(t::T, d::Dict) where T <: CategoricalTerm
    d[t.sym] = StatsModels.FullDummyCoding()
end
@inline function fill_coding_dict(t::T, d::Dict) where T <: InteractionTerm
    for i in t.terms
        d[i.sym] = StatsModels.FullDummyCoding()
    end
end
@inline function fill_coding_dict(t::T, d::Dict) where T <: Tuple
    for i in t
        if isa(i, Term)
            d[i.sym] = StatsModels.FullDummyCoding()
        else
            fill_coding_dict(i, d)
        end
    end
end
#-------------------------------------------------------------------------------


#=
"""
    G matrix
"""



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
