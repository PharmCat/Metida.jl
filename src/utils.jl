################################################################################
# Variance estimate via OLS and QR decomposition.
################################################################################
function initvar(y::Vector, X::Matrix{T}) where T
    qrx  = qr(X)
    β    = (inv(qrx.R) * qrx.Q') * y
    r    = copy(y)
    LinearAlgebra.BLAS.gemv!('N', one(T), X, β, -one(T), r)
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
tname(t::AbstractTerm) = "$(t.sym)"
tname(t::InteractionTerm) = join(tname.(t.terms), " & ")
tname(t::InterceptTerm) = "(Intercept)"
"""
    lcontrast(lmm::LMM, i::Int)

L-contrast matrix for `i` fixed effect.
"""
function lcontrast(lmm::LMM, i::Int)
    n = length(lmm.mf.f.rhs.terms)
    p = size(lmm.mm.m, 2)
    if i > n || n < 1 error("Factor number out of range 1-$(n)") end
    inds = findall(x -> x==i, lmm.mm.assign)
    if typeof(lmm.mf.f.rhs.terms[i]) <: CategoricalTerm
        mxc   = zeros(size(lmm.mf.f.rhs.terms[i].contrasts.matrix, 1), p)
        mxcv  = view(mxc, :, inds)
        mxcv .= lmm.mf.f.rhs.terms[i].contrasts.matrix
        mx    = zeros(size(lmm.mf.f.rhs.terms[i].contrasts.matrix, 1) - 1, p)
        for i = 2:size(lmm.mf.f.rhs.terms[i].contrasts.matrix, 1)
            mx[i-1, :] .= mxc[i, :] - mxc[1, :]
        end
    else
        mx = zeros(length(inds), p)
        for j = 1:length(inds)
            mx[j, inds[j]] = 1
        end
    end
    mx
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
    gmat!(G, view(lmm.result.theta, lmm.covstr.tr[r]), lmm.covstr.random[r].covtype.s)
    Symmetric(G)
end


"""
    gmatrixipd(lmm::LMM)

Return true if all variance-covariance matrix (G) of random effect is positive definite.
"""
function gmatrixipd(lmm::LMM)
    lmm.result.ipd
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
function vmatrix!(V, θ, lmm::LMM, i::Int) # pub API
    gvec = gmatvec(θ, lmm.covstr)
    zgz_base_inc!(V, gvec, θ, lmm.covstr, lmm.covstr.vcovblock[i], lmm.covstr.sblock[i])
    rmat_base_inc!(V, θ[lmm.covstr.tr[end]], lmm.covstr, lmm.covstr.vcovblock[i], lmm.covstr.sblock[i])
end

# !!! Main function REML used
function vmatrix!(V, G, θ, lmm::LMM, i::Int)
    zgz_base_inc!(V, G, θ, lmm.covstr, lmm.covstr.vcovblock[i], lmm.covstr.sblock[i])
    rmat_base_inc!(V, θ[lmm.covstr.tr[end]], lmm.covstr, lmm.covstr.vcovblock[i], lmm.covstr.sblock[i])
end

"""
    vmatrix(lmm::LMM, i::Int)

Return variance-covariance matrix V for i bolock.
"""
function vmatrix(lmm::LMM, i::Int)
    vmatrix(lmm.result.theta, lmm, i)
end

function vmatrix(θ::AbstractVector{T}, lmm::LMM, i::Int) where T
    V    = zeros(T, length(lmm.covstr.vcovblock[i]), length(lmm.covstr.vcovblock[i]))
    gvec = gmatvec(θ, lmm.covstr)
    vmatrix!(V, gvec, θ, lmm, i)
    Symmetric(V)
end

#deprecated
function vmatrix(θ::Vector, covstr::CovStructure, i::Int)
    V    = zeros(length(covstr.vcovblock[i]), length(covstr.vcovblock[i]))
    gvec = gmatvec(θ, covstr)
    zgz_base_inc!(V, gvec, θ, covstr, covstr.vcovblock[i], covstr.sblock[i])
    rmat_base_inc!(V, θ[covstr.tr[end]], covstr, covstr.vcovblock[i], covstr.sblock[i])
    Symmetric(V)
end
#=
function grad_vmatrix(θ::AbstractVector{T}, lmm::LMM, i::Int)
    V    = zeros(T, length(lmm.covstr.vcovblock[i]), length(lmm.covstr.vcovblock[i]))
    gvec = gmatvec(θ, lmm.covstr)
    vmatrix!(V, gvec, θ, lmm, i)
    Symmetric(V)
end
=#




function nblocks(lmm::LMM)
    return length(lmm.covstr.vcovblock)
end
"""
    hessian(lmm, theta)

Calculate Hessian matrix of REML for theta.
"""
function hessian(lmm, theta)
    #if !lmm.result.fit error("Model not fitted!") end
    vloptf(x) = reml_sweep_β(lmm, lmm.dv, x, lmm.result.beta)[1]
    chunk  = ForwardDiff.Chunk{min(8, length(theta))}()
    #chunk  = ForwardDiff.Chunk{1}()
    hcfg   = ForwardDiff.HessianConfig(vloptf, theta, chunk)
    ForwardDiff.hessian(vloptf, theta, hcfg)
end
function hessian(lmm)
    if !lmm.result.fit error("Model not fitted!") end
    hessian(lmm, lmm.result.theta)
end
################################################################################
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
