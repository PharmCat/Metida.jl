"""
    Make X, Z matrices and vector y for each subject;
"""
function subjblocks(df, sbj::Symbol, x::Matrix{T}, z::Matrix{T}, y::Vector{T}, rz::Matrix{T}) where T
    u = unique(df[!, sbj])
    xa  = Vector{Matrix{T}}(undef, length(u))
    za  = Vector{Matrix{T}}(undef, length(u))
    ya  = Vector{Vector{T}}(undef, length(u))
    rza = Vector{Matrix{T}}(undef, length(u))
        @inbounds @simd for i = 1:length(u)
            v = findall(x -> x == u[i], df[!, sbj])
            xa[i] = Matrix(view(x, v, :))
            za[i] = Matrix(view(z, v, :))
            ya[i] = Vector(view(y, v))
            rza[i] = Matrix(view(rz, v, :))
        end
    return xa, za, rza, ya
end
function subjblocks(df, sbj::Symbol, x::Matrix{T}, z::Matrix{T}, y::Vector{T}, rz::Nothing) where T
    u = unique(df[!, sbj])
    xa  = Vector{Matrix{T}}(undef, length(u))
    za  = Vector{Matrix{T}}(undef, length(u))
    ya  = Vector{Vector{T}}(undef, length(u))
    rza = Vector{Matrix{T}}(undef, length(u))
        @inbounds @simd for i = 1:length(u)
            v = findall(x -> x == u[i], df[!, sbj])
            xa[i]  = Matrix(view(x, v, :))
            za[i]  = Matrix(view(z, v, :))
            ya[i]  = Vector(view(y, v))
            rza[i] = Matrix{T}(undef, 0, 0)
        end
    return xa, za, rza, ya
end
function subjblocks(df, sbj)
    if isa(sbj, Symbol)
        u = unique(df[!, sbj])
        r = Vector{Vector{Int}}(undef, length(u))
        @inbounds @simd for i = 1:length(u)
            r[i] = findall(x -> x == u[i], df[!, sbj])
        end
        return r
    end
    r = Vector{Vector{Int}}(undef, 1)
    r[1] = collect(1:size(df, 1))
    r
end
"""
Variance estimate via OLS and QR decomposition.
"""
function initvar(y::Vector, X::Matrix{T}) where T
    qrx  = qr(X)
    β    = inv(qrx.R) * qrx.Q' * y
    r    = y .- X * β
    sum(x -> x * x, r)/(length(r) - size(X, 2)), β
end

################################################################################
#                        VAR LINK
################################################################################

function vlink(σ::T) where T <: Real
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

function rholinksigmoid2(ρ::T) where T <: Real
    return atan(ρ)/pi*2.0
end
function rholinksigmoid2r(ρ::T) where T <: Real
    return tan(ρ*pi/2.0)
end

################################################################################

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
function varlinkvecapply(v, f)
    rv = similar(v)
    for i = 1:length(v)
        rv[i] = f[i](v[i])
    end
    rv
end
function varlinkvecapply!(v, f)
    for i = 1:length(v)
        v[i] = f[i](v[i])
    end
    v
end

################################################################################

function vmatr(lmm, i)
    θ  = lmm.result.theta
    G  = gmat_base(θ, lmm.covstr)
    V  = mulαβαt(view(lmm.data.zv, lmm.data.block[i],:), G)
    if length(lmm.data.zrv) > 0
        rmat_basep!(V, θ[lmm.covstr.tr[end]], view(lmm.data.zrv, lmm.data.block[i],:), lmm.covstr)
    else
        rmat_basep!(V, θ[lmm.covstr.tr[end]], lmm.data.zrv, lmm.covstr)
    end
    V
end

function gmatr(lmm, i)
    θ  = lmm.result.theta
    gmat_base(θ, lmm.covstr)
end

################################################################################

function m2logreml(lmm)
    lmm.result.reml
end
function logreml(lmm)
    -m2logreml(lmm)/2.
end
