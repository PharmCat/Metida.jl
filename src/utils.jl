################################################################################
# Variance estimate via OLS and QR decomposition.
################################################################################
function initvar(y::Vector, X::Matrix{T}) where T
    qrx  = qr(X)
    β    = (inv(qrx.R) * qrx.Q') * y
    r    = copy(y)
    LinearAlgebra.BLAS.gemv!('N', 1.0, X, β, -1.0, r)
    #r    = y .- X * β
    sum(x -> x * x, r)/(length(r) - size(X, 2)), β
end
################################################################################
function nterms(lmm::LMM)
    nterms(lmm.mf)
end
function nterms(mf::ModelFrame)
    mf.schema.schema.count
end
function nterms(rhs::Union{Tuple{Vararg{AbstractTerm}}, Nothing, AbstractTerm})
    if isa(rhs, Term)
        p = 1
    elseif isa(rhs, Tuple)
        p = length(rhs)
    else
        p = 0
    end
    p
end
################################################################################
#                        VAR LINK
################################################################################

function vlink(σ::T) where T <: Real
    if σ < -21.0 return one(T)*7.582560427911907e-10 end #Experimental
    exp(σ)
end
function vlinkr(σ::T) where T <: Real
    log(σ)
end

function vlinksq(σ::T) where T <: Real
    σ*σ
end
function vlinksqr(σ::T) where T <: Real
    sqrt(σ)
end

function rholinkpsigmoid(ρ::T) where T <: Real
    return 1.0/(1.0 + exp(-ρ * 0.5))
end
function rholinkpsigmoidr(ρ::T) where T <: Real
    return - log(1.0/ρ - 1.0) / 0.5
end

function rholinksigmoid(ρ::T) where T <: Real
    return 1.0/(1.0 + exp(- ρ * 0.1)) * 2.0 - 1.0
end
function rholinksigmoidr(ρ::T) where T <: Real
    return - log(1.0/(ρ+1.0)*2.0 - 1.0) / 0.1
end

function rholinksqsigmoid(ρ::T) where T <: Real
    return ρ/sqrt(1.0 + (ρ)^2)
end
function rholinksqsigmoidr(ρ::T) where T <: Real
    return sign(ρ)*sqrt(ρ^2/(1.0 - ρ^2))
end

function rholinksigmoidatan(ρ::T) where T <: Real
    return atan(ρ)/pi*2.0
end
function rholinksigmoidatanr(ρ::T) where T <: Real
    return tan(ρ*pi/2.0)
end

################################################################################
################################################################################
function varlinkvecapply!(v, p; varlinkf = :exp, rholinkf = :sigm)
    @inbounds for i = 1:length(v)
        if p[i] == :var
            if varlinkf == :exp
                v[i] = vlink(v[i])
            elseif varlinkf == :sq
                v[i] = vlinksq(v[i])
            elseif varlinkf == :identity
                v[i] = identity(v[i])
            end
        elseif p[i] == :rho
            if rholinkf == :sigm
                v[i] = rholinksigmoid(v[i])
            elseif rholinkf == :atan
                v[i] = rholinksigmoidatan(v[i])
            elseif rholinkf == :sqsigm
                v[i] = rholinksqsigmoid(v[i])
            elseif rholinkf == :psigm
                v[i] = rholinkpsigmoid(v[i])
            end
        end
    end
    v
end
function varlinkrvecapply!(v, p; varlinkf = :exp, rholinkf = :sigm)
    @inbounds for i = 1:length(v)
        if p[i] == :var
            if varlinkf == :exp
                v[i] = vlinkr(v[i])
            elseif varlinkf == :sq
                v[i] = vlinksqr(v[i])
            elseif varlinkf == :identity
                v[i] = identity(v[i])
            end
        elseif p[i] == :rho
            if rholinkf == :sigm
                v[i] = rholinksigmoidr(v[i])
            elseif rholinkf == :atan
                v[i] = rholinksigmoidatanr(v[i])
            elseif rholinkf == :sqsigm
                v[i] = rholinksqsigmoidr(v[i])
            elseif rholinkf == :psigm
                v[i] = rholinkpsigmoidr(v[i])
            end
        end
    end
    v
end
function varlinkvecapply(v, p; varlinkf = :exp, rholinkf = :sigm)
    s = similar(v)
    @inbounds for i = 1:length(v)
        if p[i] == :var
            if varlinkf == :exp
                s[i] = vlink(v[i])
            elseif varlinkf == :sq
                s[i] = vlinksq(v[i])
            elseif varlinkf == :identity
                s[i] = identity(v[i])
            end
        elseif p[i] == :rho
            if rholinkf == :sigm
                s[i] = rholinksigmoid(v[i])
            elseif rholinkf == :atan
                s[i] = rholinksigmoidatan(v[i])
            elseif rholinkf == :sqsigm
                s[i] = rholinksqsigmoid(v[i])
            elseif rholinkf == :psigm
                s[i] = rholinkpsigmoid(v[i])
            end
        else
            s[i] = v[i]
        end
    end
    s
end
################################################################################
function m2logreml(lmm)
    lmm.result.reml
end
function logreml(lmm)
    -m2logreml(lmm)/2
end
function m2logreml(lmm, theta)
    reml_sweep_β(lmm, LMMDataViews(lmm), theta)[1]
end
################################################################################

function optim_callback(os)
    false
end
################################################################################
"""
    gmatrix(lmm::LMM{T}, r::Int) where T
"""
function gmatrix(lmm::LMM{T}, r::Int) where T
    if isnothing(lmm.result.theta) error("No results or model not fitted!") end
    if r > length(lmm.covstr.random) error("Invalid random effect number: $(r)!") end
    G = zeros(T, lmm.covstr.q[r], lmm.covstr.q[r])
    gmat_switch!(G, lmm.result.theta, lmm.covstr, r)
    Symmetric(G)
end
"""
    rmatrix(lmm::LMM{T}, i::Int) where T
"""
function rmatrix(lmm::LMM{T}, i::Int) where T
    if !lmm.result.fit error("Model not fitted!") end
    if i > length(lmm.covstr.vcovblock) error("Invalid block number: $(i)!") end
    q    = length(lmm.covstr.vcovblock[i])
    R    = zeros(T, q, q)
    rmat_base_inc!(R, lmm.result.theta[lmm.covstr.tr[end]], lmm.covstr, lmm.covstr.vcovblock[i], lmm.covstr.sblock[i])
    Symmetric(R)
end
"""
    vmatrix!(V, θ, lmm, i)

Update variance-covariance matrix V for i bolock. Upper triangular updated.
"""
function vmatrix!(V, θ, lmm, i)
    zgz_base_inc!(V, θ, lmm.covstr, lmm.covstr.vcovblock[i], lmm.covstr.sblock[i])
    rmat_base_inc!(V, θ[lmm.covstr.tr[end]], lmm.covstr, lmm.covstr.vcovblock[i], lmm.covstr.sblock[i])
end
function vmatrix(θ, lmm, i)
    V = zeros(length(lmm.covstr.vcovblock[i]), length(lmm.covstr.vcovblock[i]))
    vmatrix!(V, θ, lmm, i)
    Symmetric(V)
end
function vmatrix(lmm, i)
    vmatrix(lmm.result.theta, lmm, i)
end

function nblocks(lmm)
    return length(lmm.covstr.vcovblock)
end
"""
    hessian(lmm, theta)

Calculate Hessian matrix of REML for theta.
"""
function hessian(lmm, theta)
    #if !lmm.result.fit error("Model not fitted!") end
    vloptf(x) = reml_sweep_β(lmm, x, lmm.result.beta)[1]
    chunk  = ForwardDiff.Chunk{min(10, length(theta))}()
    hcfg   = ForwardDiff.HessianConfig(vloptf, theta, chunk)
    ForwardDiff.hessian(vloptf, theta, hcfg)
end
function hessian(lmm)
    if !lmm.result.fit error("Model not fitted!") end
    hessian(lmm, lmm.result.theta)
end
################################################################################

function get_symb(t::T; v = Vector{Symbol}(undef, 0)) where T <: Union{ConstantTerm, InterceptTerm, FunctionTerm}
    v
end
function get_symb(t::T; v = Vector{Symbol}(undef, 0)) where T <: Union{Term, CategoricalTerm}
    push!(v, t.sym)
    v
end
function get_symb(t::T; v = Vector{Symbol}(undef, 0)) where T <: InteractionTerm
    for i in t.terms
        get_symb(i; v = v)
    end
    v
end
function get_symb(t::T; v = Vector{Symbol}(undef, 0)) where T <: Tuple{Vararg{AbstractTerm}}
    for i in t
        get_symb(i; v = v)
    end
    v
end


################################################################################
# logdet with check
#=
function logdet_(C::Cholesky, noerror)
    dd = zero(real(eltype(C)))
    @inbounds for i in 1:size(C.factors,1)
        v = real(C.factors[i,i])
        if v > 0
            dd += log(v)
        else
            C.factors[i,i] = LOGDETCORR
            dd += log(real(C.factors[i,i]))
            noerror = false
        end
    end
    dd + dd, noerror
end
=#

function StatsModels.termvars(ve::VarEffect)
    termvars(ve.formula)
end
function StatsModels.termvars(ve::Vector{VarEffect})
    union(termvars.(ve)...)
end

################################################################################

"""
    rand(rng::AbstractRNG, lmm::LMM{T}) where T

Generate random responce vector for fitted 'lmm' model.
"""
function rand(rng::AbstractRNG, lmm::LMM{T}) where T
    if !lmm.result.fit error("Model not fitted!") end
    rand(rng::AbstractRNG, lmm::LMM{T}, lmm.result.theta, lmm.result.beta)
end

"""
    rand(rng::AbstractRNG, lmm::LMM{T}; theta) where T

!!! warning
    Experimental

Generate random responce vector 'lmm' model, theta covariance vector, and zero means.
"""
function rand(rng::AbstractRNG, lmm::LMM{T}, theta) where T
    n = length(lmm.covstr.vcovblock)
    v = Vector{Float64}(undef, nobs(lmm))
    for i = 1:n
        q    = length(lmm.covstr.vcovblock[i])
        V    = zeros(q, q)
        Metida.vmatrix!(V, theta, lmm, i)
        copyto!(view(v, lmm.covstr.vcovblock[i]), rand(rng, MvNormal(Symmetric(V))))
    end
    v
end
rand(lmm::LMM, theta) = rand(default_rng(), lmm, theta)
"""
    rand(rng::AbstractRNG, lmm::LMM{T}; theta, beta) where T

Generate random responce vector 'lmm' model, theta covariance vector and mean's vector.
"""
function rand(rng::AbstractRNG, lmm::LMM{T}, theta, beta) where T
    if length(beta) != size(lmm.data.xv, 2) error("Wrong beta length!") end
    n = length(lmm.covstr.vcovblock)
    v = Vector{Float64}(undef, nobs(lmm))
    for i = 1:n
        q    = length(lmm.covstr.vcovblock[i])
        m    = lmm.dv.xv[i] * beta
        V    = zeros(q, q)
        Metida.vmatrix!(V, theta, lmm, i)
        copyto!(view(v, lmm.covstr.vcovblock[i]), rand(rng, MvNormal(m, Symmetric(V))))
    end
    v
end
rand(lmm::LMM, theta, beta) = rand(default_rng(), lmm, theta, beta)
