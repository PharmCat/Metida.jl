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
function HeterogeneousCompoundSymmetry()
    CovarianceType(:CSH, ffxpone, ffx, ffxone)
end
const CSH = HeterogeneousCompoundSymmetry()
################################################################################
#                  EFFECT
################################################################################
struct VarEffect
    model::Union{Tuple{Vararg{AbstractTerm}}, Nothing}
    covtype::CovarianceType
    coding::Dict{Symbol, AbstractContrasts}
    function VarEffect(model, covtype::T, coding) where T <: AbstractCovarianceType
        if coding === nothing && model !== nothing
            coding = Dict{Symbol, AbstractContrasts}()
            #fill_coding_dict(model, coding)
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
################################################################################
#                            COVARIANCE STRUCTURE
################################################################################
struct CovStructure{T} <: AbstractCovarianceStructure
    random::Vector{VarEffect}                                                   #Random effects
    repeated::VarEffect                                                         #Repearted effects
    schema::Vector{Tuple}
    rcnames::Vector{String}
    z::Matrix{T}                                                                #Z matrix
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
        z       = Matrix{Float64}(undef, size(data, 1), 0)

        #rschema = apply_schema(random[1].model, StatsModels.schema(data, random[1].coding))
        #z       = reduce(hcat, modelcols(rschema, data))
        #schema[1] = rschema
        #q[1]    = size(z, 2)
        #t[1]    = random[1].covtype.f(q[1])
        #tr[1]   = UnitRange(1, t[1])
        #if length(random) > 1
            for i = 1:length(random)
                if length(random[i].coding) == 0
                    fill_coding_dict!(random[i].model, random[i].coding, data)
                end
                rschema = apply_schema(random[i].model, StatsModels.schema(data, random[i].coding))
                ztemp = reduce(hcat, modelcols(rschema, data))
                schema[i] = rschema
                q[i]    = size(ztemp, 2)
                t[i]    = random[i].covtype.f(q[i])
                z       = hcat(z, ztemp)
                if i > 1
                    tr[i]   = UnitRange(sum(t[1:i-1]) + 1, sum(t[1:i-1])+t[i])
                else
                    tr[1]   = UnitRange(1, t[1])
                end
            end
        #end
        if repeated.model !== nothing
            fill_coding_dict!(repeated.model, repeated.coding, data)
            rschema = apply_schema(repeated.model, StatsModels.schema(data, repeated.coding))
            rz = reduce(hcat, modelcols(rschema, data))
            schema[end] = rschema
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
        new{eltype(z)}(random, repeated, schema, rcnames, z, rz, q, t, tr, tl, ct)
    end
end
################################################################################
@inline function schemalength(s)
    if isa(s, Tuple)
        return length(s)
    else
        return 1
    end
end
################################################################################
#                       G MATRIX FUNCTIONS
################################################################################
@inline function gmat_base(θ::Vector{T}, covstr) where T
    q = size(covstr.z, 2)
    mx = zeros(T, q, q)
    for i = 1:length(covstr.random)
        s = 1 + sum(covstr.q[1:i]) - covstr.q[i]
        e = sum(covstr.q[1:i])
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
    end
    mx
end
################################################################################
@inline function gmat_si!(mx, θ::Vector{T}, zn::Int, ::CovarianceType) where T
    val = θ[1] ^ 2
    for i = 1:size(mx, 1)
        mx[i, i] = val
    end
    nothing
end
@inline function gmat_vc!(mx, θ::Vector{T}, ::Int, ::CovarianceType) where T
    for i = 1:size(mx, 1)
        mx[i, i] = θ[i]
    end
    nothing
end
@inline function gmat_ar!(mx, θ::Vector{T}, zn::Int, ::CovarianceType) where T
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
@inline function gmat_arh!(mx, θ::Vector{T}, ::Int, ::CovarianceType) where T
    s = size(mx, 1)
    for m = 1:s
        @inbounds mx[m, m] = θ[m]
    end
    if s > 1
        for m = 1:s - 1
            for n = m + 1:s
                @inbounds mx[m, n] = mx[m, m] * mx[n, n] * θ[end] ^ (n - m)
                @inbounds mx[n, m] = mx[m, n]
            end
        end
    end
    for m = 1:s
        @inbounds mx[m, m] = mx[m, m] * mx[m, m]
    end
    nothing
end
@inline function gmat_csh!(mx, θ::Vector{T}, ::Int, ::CovarianceType) where T
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
################################################################################
#                         R MATRIX FUNCTIONS
################################################################################
@inline function rmat_basep!(mx, θ::AbstractVector{T}, zrv, covstr) where T
    if covstr.repeated.covtype.s == :SI
        rmatp_si!(mx, θ, zrv, covstr.repeated.covtype)
    elseif covstr.repeated.covtype.s == :VC
        rmatp_vc!(mx, θ, zrv, covstr.repeated.covtype)
    elseif covstr.repeated.covtype.s == :AR
        rmatp_ar!(mx, θ, zrv, covstr.repeated.covtype)
    elseif covstr.repeated.covtype.s == :ARH
        rmatp_arh!(mx, θ, zrv, covstr.repeated.covtype)
    elseif covstr.repeated.covtype.s == :CSH
        rmatp_csh!(mx, θ, zrv, covstr.repeated.covtype)
    else
        throw(ErrorException("Unknown covariance structure: $(covstr.repeated.covtype.s)"))
    end
end
@inline function rmatp_si!(mx, θ::Vector{T}, ::Matrix, ::CovarianceType) where T
    θsq = θ[1]*θ[1]
    for i = 1:size(mx, 1)
            mx[i, i] += θsq
    end
    nothing
end
@inline function rmatp_vc!(mx, θ::Vector{T}, rz,  ::CovarianceType) where T
    for i = 1:size(mx, 1)
        for c = 1:length(θ)
            mx[i, i] += θ[c]*θ[c]*rz[i, c]
        end
    end
    nothing
end
@inline function rmatp_ar!(mx, θ::Vector{T}, rz, ::CovarianceType) where T
    rn  = size(mx, 1)
    mx  = Matrix{T}(undef, rn, rn)
    de  = θ[1] ^ 2
    for m = 1:rn
        mx[m, m] += de
    end
    if rn > 1
        for m = 1:rn - 1
            for n = m + 1:rn
                ode = de * θ[2] ^ (n - m)
                @inbounds mx[m, n] += ode
                @inbounds mx[n, m] = mx[m, n]
            end
        end
    end
    nothing
end
@inline function rmatp_arh!(mx, θ::Vector{T}, rz, ::CovarianceType) where T
    vec   = rz * (θ[1:end-1])
    rn    = size(mx, 1)
    if rn > 1
        for m = 1:rn - 1
            for n = m + 1:rn
                @inbounds mx[m, n] += vec[m] * vec[n] * θ[end] ^ (n - m)
            end
        end
    end
    for m = 1:rn
        @inbounds mx[m, m] += vec[m] * vec[m]
    end
    nothing
end
@inline function rmatp_csh!(mx, θ::Vector{T}, rz, ::CovarianceType) where T
    vec   = rz * (θ[1:end-1])
    rn    = size(mx, 1)
    if rn > 1
        for m = 1:rn - 1
            for n = m + 1:rn
                @inbounds mx[m, n] += vec[m] * vec[n] * θ[end]
            end
        end
    end
    for m = 1:rn
        @inbounds mx[m, m] += vec[m] * vec[m]
    end
    nothing
end
################################################################################
#                            CONTRAST CODING
################################################################################

ContinuousTerm

@inline function fill_coding_dict!(t::T, d::Dict, data) where T <: ConstantTerm
end
@inline function fill_coding_dict!(t::T, d::Dict, data) where T <: Term
    if typeof(data[!, t.sym]) <: CategoricalArray
        d[t.sym] = StatsModels.FullDummyCoding()
    end
end
@inline function fill_coding_dict!(t::T, d::Dict, data) where T <: CategoricalTerm
    if typeof(data[!, t.sym])  <: CategoricalArray
        d[t.sym] = StatsModels.FullDummyCoding()
    end
end
@inline function fill_coding_dict!(t::T, d::Dict, data) where T <: InteractionTerm
    for i in t.terms
        if typeof(data[!, i.sym])  <: CategoricalArray
            d[i.sym] = StatsModels.FullDummyCoding()
        end
    end
end
@inline function fill_coding_dict!(t::T, d::Dict, data) where T <: Tuple
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

################################################################################
