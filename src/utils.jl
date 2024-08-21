################################################################################
# Variance estimate via OLS and QR decomposition.
################################################################################
function initvar(y::Vector, X::Matrix{T}) where T
    qrx  = qr(X)
    β    = (inv(qrx.R) * qrx.Q') * y
    r    = copy(y)
    LinearAlgebra.BLAS.gemv!('N', one(T), X, β, -one(T), r)
    #r    = y .- X * β
    dot(r, r)/(length(r) - size(X, 2)), β
end
################################################################################
function fixedeffn(f::FormulaTerm)
    length(f.rhs.terms) - !StatsModels.hasintercept(f)
end
function fixedeffn(lmm::LMM)
    fixedeffn(lmm.f) 
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
"""
    Term name.
"""
tname(t::AbstractTerm) = "$(t.sym)"
tname(t::InteractionTerm) = join(tname.(t.terms), " & ")
tname(t::InterceptTerm) = "(Intercept)"
function tname(t::FunctionTerm) 
    args = string(t.args_parsed[1])
    if length(t.args_parsed) > 1
        for i = 2:length(t.args_parsed)
            args *= ", "*string(t.args_parsed[i])
        end
    end
    string(t.forig)*"("*args*")"
end


"""
    lcontrast(lmm::LMM, i::Int)

L-contrast matrix for `i` fixed effect.
"""
function lcontrast(lmm::LMM, i::Int)
    n = length(lmm.f.rhs.terms)
    p = size(lmm.data.xv, 2)
    if i > n || n < 1 error("Factor number out of range 1-$(n)") end
    inds = findall(x -> x==i, assign(lmm))
    if typeof(lmm.f.rhs.terms[i]) <: CategoricalTerm
        mxc   = zeros(size(lmm.f.rhs.terms[i].contrasts.matrix, 1), p)
        mxcv  = view(mxc, :, inds)
        mxcv .= lmm.f.rhs.terms[i].contrasts.matrix
        mx    = zeros(size(lmm.f.rhs.terms[i].contrasts.matrix, 1) - 1, p)
        for i = 2:size(lmm.f.rhs.terms[i].contrasts.matrix, 1)
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
    return 1/(1 + exp(-ρ / 2))
end
function rholinkpsigmoidr(ρ::T) where T <: Real
    return - log(1/ρ - 1) * 2
end

function rholinksigmoid(ρ::T) where T <: Real
    return 1/(1 + exp(- ρ * 0.1)) * 2 - 1
end
function rholinksigmoidr(ρ::T) where T <: Real
    return - log(1/(ρ + 1) * 2 - 1) / 0.1
end

function rholinksqsigmoid(ρ::T) where T <: Real
    return ρ/sqrt(1 + (ρ)^2)
end
function rholinksqsigmoidr(ρ::T) where T <: Real
    return sign(ρ)*sqrt(ρ^2/(1 - ρ^2))
end

function rholinksigmoidatan(ρ::T) where T <: Real
    return atan(ρ) / pi * 2
end
function rholinksigmoidatanr(ρ::T) where T <: Real
    return tan(ρ * pi / 2)
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
function m2logreml(lmm, theta; maxthreads::Int = num_cores())
    reml_sweep_β(lmm, LMMDataViews(lmm), theta; maxthreads = maxthreads)[1]
end
################################################################################

function optim_callback(os)
    false
end
################################################################################
"""
    zeroeff(eff)

Return true if CovarianceType is ZERO.
"""
function zeroeff(eff)
    isa(eff.covtype.s, ZERO)
end
"""
    raneffn(lmm)
Retuen number of random effects.
"""
function raneffn(lmm)
    if zeroeff(lmm.covstr.random[1]) 
        return 0
    end
    length(lmm.covstr.random)
end

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
    rθ   = lmm.covstr.tr[lmm.covstr.rn + 1:end]
    rmat_base_inc!(R, lmm.result.theta, rθ, lmm.covstr, i)
    Symmetric(R)
end

#####################################################################

function applywts!(::Any, ::Int, ::Nothing)
    nothing
end

function applywts!(V::AbstractMatrix, i::Int, wts::LMMWts{<:Vector})
    mulβdαβd!(V, wts.sqrtwts[i])
end

function applywts!(V::AbstractMatrix, i::Int, wts::LMMWts{<:Matrix})
    V .*= wts.sqrtwts[i]
end

#####################################################################

#####################################################################
"""
    vmatrix!(V, θ, lmm, i)

Update variance-covariance matrix V for i bolock. Upper triangular updated.
"""
function vmatrix!(V, θ, lmm::LMM, i::Int) # pub API
    gvec = gmatvec(θ, lmm.covstr)
    rθ   = lmm.covstr.tr[lmm.covstr.rn + 1:end]
    rmat_base_inc!(V, θ, rθ, lmm.covstr, i) # Repeated vector
    applywts!(V, i, lmm.wts) 
    zgz_base_inc!(V, gvec, lmm.covstr, i)
    
end

# !!! Main function REML used
@noinline function vmatrix!(V, G, θ, rθ, lmm::LMM, i::Int)
    rmat_base_inc!(V, θ, rθ, lmm.covstr, i)  # Repeated vector
    applywts!(V, i, lmm.wts) 
    zgz_base_inc!(V, G, lmm.covstr, i)
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
    rθ   = lmm.covstr.tr[lmm.covstr.rn + 1:end]
    vmatrix!(V, gvec, θ, rθ, lmm, i) # Repeated vector
    Symmetric(V)
end
# For Multiple Imputation
function vmatrix(θ::Vector, covstr::CovStructure, lmmwts, i::Int)
    V    = zeros(length(covstr.vcovblock[i]), length(covstr.vcovblock[i]))
    gvec = gmatvec(θ, covstr)
    rθ   = covstr.tr[covstr.rn + 1:end]
    rmat_base_inc!(V, θ, rθ, covstr, i)  # Repeated vector
    applywts!(V, i, lmmwts) 
    zgz_base_inc!(V, gvec, covstr, i)
    Symmetric(V)
end

function blockgmatrix(lmm::LMM{T}) where T
    blockgmatrix(lmm, (1, 1))
end

function blockgmatrix(lmm::LMM{T}, v) where T
    if raneffn(lmm) == 0 return nothing end
    p = 0
    for i = 1:raneffn(lmm)
        p += lmm.covstr.q[i]*v[i]
    end
    G = zeros(T, p, p)
    s = 1
    for i = 1:raneffn(lmm)
        g = gmatrix(lmm, i)
        for j = 1:v[i]
            e = s + lmm.covstr.q[i]-1
            G[s:e, s:e] .= g
            s = e + 1
        end
    end
    G
end

function blockzmatrix(lmm::LMM{T}, i) where T
    sn = raneflenv(lmm.covstr, i)
    mx = Vector{Matrix{T}}(undef, raneffn(lmm))
    s = 1
    for j = 1:raneffn(lmm)
        e  = s + lmm.covstr.q[j] - 1
        obsn = length(lmm.covstr.vcovblock[i])
        zv   = view(lmm.covstr.z, lmm.covstr.vcovblock[i], s:e) 
        smx  = zeros(T, obsn, length(lmm.covstr.esb.sblock[i,j])*lmm.covstr.q[j])
        for k = 1:length(lmm.covstr.esb.sblock[i,j])
            s1 = 1 + (k - 1) * lmm.covstr.q[j]
            e1 = s1 + lmm.covstr.q[j] - 1
            smx[lmm.covstr.esb.sblock[i,j][k][1], s1:e1] .= view(zv, lmm.covstr.esb.sblock[i,j][k][1], :)
        end
        s  = e + 1
        mx[j] = smx
    end
   hcat(mx...)
end
"""
    raneff(lmm::LMM{T}, i)

Vector of random effect coefficients for block `i`.
"""
function raneff(lmm::LMM{T}, block) where T
    if raneffn(lmm) == 0 return nothing end
    sn = raneflenv(lmm.covstr, block)
    G  = blockgmatrix(lmm, sn)
    Z  = blockzmatrix(lmm, block)
    rv = G * Z' * inv(vmatrix(lmm, block)) * (lmm.dv.yv[block] - lmm.dv.xv[block]*lmm.result.beta)
    
    rvsbj = Vector{Vector}(undef, length(sn))
    for i = 1:raneffn(lmm)
        rvsbj[i] = Vector{Pair}(undef, sn[i])
        st = 1
        if i > 1
            for j = 1:i-1
                st += sn[j] * lmm.covstr.q[j]
            end
        end
        for j = 1:sn[i]
            s = st + (j - 1) * lmm.covstr.q[i]
            e = s  + lmm.covstr.q[i] - 1
            sbnn     = getsubjnn(lmm.covstr, i, block, j)
            subjname = getsubjname(lmm.covstr, sbnn)
            rvsbj[i][j] =  subjname => rv[s:e]
        end
    end
    rvsbj
    
end
"""
    raneff(lmm::LMM{T})

Vector of random effect coefficients for all subjects by each random effect.
"""
function raneff(lmm::LMM)
    fb = raneff(lmm, 1)
    n = nblocks(lmm)
    if n > 1
        for i = 2:n
            sb = raneff(lmm, i)
            for j = 1:raneffn(lmm)
                append!(fb[j], sb[j])
            end
        end
    end
    fb
end

"""
    Number of blocks
"""
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
    hcfg   = ForwardDiff.HessianConfig(vloptf, theta, chunk)
    ForwardDiff.hessian(vloptf, theta, hcfg)
end
function hessian(lmm)
    if !lmm.result.fit error("Model not fitted!") end
    hessian(lmm, lmm.result.theta)
end
################################################################################
################################################################################
function StatsModels.termvars(ve::VarEffect)
    termvars(ve.formula)
end
function StatsModels.termvars(ve::Vector{VarEffect})
    union(termvars.(ve)...)
end

################################################################################
#=
asgn(f::FormulaTerm) = asgn(f.rhs)
asgn(t) = mapreduce(((i,t), ) -> i*ones(StatsModels.width(t)),
                    append!,
                    enumerate(StatsModels.vectorize(t)),
                    init=Int[])
=#