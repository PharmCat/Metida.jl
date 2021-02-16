################################################################################
# Intersect dataframe.
################################################################################
#=
function intersectdf(df, s)::Vector
    if isa(s, Nothing) return [collect(1:size(df, 1))] end
    if isa(s, Symbol) s = [s] end
    if length(s) == 0 return [collect(1:size(df, 1))] end
    u   = unique(@view df[:, s])
    sort!(u, s)
    res = Vector{Vector{Int}}(undef, size(u, 1))
    v   = Vector{Dict{}}(undef, size(u, 2))
    v2  = Vector{Vector{Int}}(undef, size(u, 2))
    for i2 = 1:size(u, 2)
        uv = unique(@view u[:, i2])
        v[i2] = Dict{Any, Vector{Int}}()
        for i = 1:length(uv)
            v[i2][uv[i]] = findall(x -> x == uv[i], @view df[:,  s[i2]])
        end
    end
    for i2 = 1:size(u, 1)
        for i = 1:length(s)
            v2[i] = v[i][u[i2, i]]
        end
        res[i2] = collect(intersect(Set.(v2)...))
        #res[i2] = intersect(v2...)
        sort!(res[i2])
    end
    res
end
################################################################################
# Intersect subject set in effects.
################################################################################
function intersectsubj(random, repeated)
    a  = Vector{Vector{Symbol}}(undef, length(random)+1)
    eq = true
    for i = 1:length(random)
        a[i] = random[i].subj
    end
    a[end] = repeated.subj
    for i = 2:length(a)
        if !(issetequal(a[1], a[i]))
            eq = false
            break
        end
    end
    intersect(a...), eq
end
function intersectsubj(random)
    if !isa(random, Vector) return random.subj end
    a  = Vector{Vector{Symbol}}(undef, length(random))
    for i = 1:length(random)
        a[i] = random[i].subj
    end
    intersect(a...)
end
=#
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
    return 1.0/(1.0 + exp(ρ))
end
function rholinkpsigmoidr(ρ::T) where T <: Real
    return log(1.0/ρ - 1.0)
end

function rholinksigmoid(ρ::T) where T <: Real
    return ρ/sqrt(1.0 + ρ^2)
end
function rholinksigmoidr(ρ::T) where T <: Real
    return sign(ρ)*sqrt(ρ^2/(1.0 - ρ^2))
end

function rholinksigmoidatan(ρ::T) where T <: Real
    return atan(ρ)/pi*2.0
end
function rholinksigmoidatanr(ρ::T) where T <: Real
    return tan(ρ*pi/2.0)
end

################################################################################
#=
function varlinkvec(v)
    fv = Vector{Function}(undef, length(v))
    for i = 1:length(v)
        if v[i] == :var fv[i] = vlink else fv[i] = rholinksigmoid end
    end
    fv
end
function varlinkrvec(v)
    fv = Vector{Function}(undef, length(v))
    for i = 1:length(v)
        if v[i] == :var fv[i] = vlinkr else fv[i] = rholinksigmoidr end
    end
    fv
end
=#
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
    if !lmm.result.fit error("Model not fitted!") end
    if r > length(lmm.covstr.random) error("Invalid random effect number: $(r)!") end
    G = zeros(T, lmm.covstr.q[r], lmm.covstr.q[r])
    gmat_switch!(G, lmm.result.theta, lmm.covstr, r)
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
    V
end
function vmatrix(lmm, i)
    V = vmatrix(lmm.result.theta, lmm, i)
    V
end

function nblocks(lmm)
    return length.covstr.vcovblock
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
