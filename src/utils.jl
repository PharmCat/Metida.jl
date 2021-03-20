################################################################################
# Variance estimate via OLS and QR decomposition.
################################################################################
function initvar(y::Vector, X::Matrix{T}) where T
    qrx  = qr(X)
    β    = inv(qrx.R) * qrx.Q' * y
    r    = y .- X * β
    sum(x -> x * x, r)/(length(r) - size(X, 2)), β
end
################################################################################
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
    for i = 1:length(v)
        if p[i] == :var
            v[i] = vlink(v[i])
        else
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
    for i = 1:length(v)
        if p[i] == :var
            v[i] = vlinkr(v[i])
        else
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
    for i = 1:length(v)
        if p[i] == :var
            s[i] = vlink(v[i])
        else
            if rholinkf == :sigm
                s[i] = rholinksigmoid(v[i])
            elseif rholinkf == :atan
                s[i] = rholinksigmoidatan(v[i])
            elseif rholinkf == :sqsigm
                s[i] = rholinksqsigmoid(v[i])
            elseif rholinkf == :psigm
                s[i] = rholinkpsigmoid(v[i])
            end
        end
    end
    s
end
################################################################################
function m2logreml(lmm)
    lmm.result.reml
end
function logreml(lmm)
    -m2logreml(lmm)/2.
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

Update variance-covariance matrix V for i bolock.
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
    chunk  = ForwardDiff.Chunk{1}()
    hcfg   = ForwardDiff.HessianConfig(vloptf, theta, chunk)
    ForwardDiff.hessian(vloptf, theta, hcfg, Val{false}())
end
function hessian(lmm)
    if !lmm.result.fit error("Model not fitted!") end
    hessian(lmm, lmm.result.theta)
end
