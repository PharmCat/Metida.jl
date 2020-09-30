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

"""
Variance estimate via OLS and QR decomposition.
"""
@inline function initvar(y::Vector, X::Matrix{T}) where T
    qrx  = qr(X)
    β    = inv(qrx.R) * qrx.Q' * y
    r    = y - X * b
    sum(x -> x * x, r)/(length(r) - size(X, 2)), β
end

################################################################################
#                        VAR LINK
################################################################################

@inline function vlink(σ::T) where T <: Real
    exp(σ)
end
@inline function vlinkr(σ::T) where T <: Real
    log(σ)
end

@inline function rholinkpsigmoid(ρ::T) where T <: Real
    return 1.0/(1.0 + exp(ρ))
end
@inline function rholinkpsigmoidr(ρ::T) where T <: Real
    return log(1.0/ρ - 1.0)
end

@inline function rholinksigmoid(ρ::T, m) where T <: Real
    return ρ/sqrt(1.0 + ρ^2)
end
@inline function rholinksigmoidr(ρ::T, m) where T <: Real
    return sign(ρ)*sqrt(ρ^2/(1.0 - ρ^2))
end

@inline function rholinksigmoid2(ρ::T, m) where T <: Real
    return atan(ρ)/pi*2.0
end
@inline function rholinksigmoid2r(ρ::T, m) where T <: Real
    return tan(ρ*pi/2.0)
end

################################################################################

@inline function varlinkvec(v)
    fv = Vector{Function}(undef, length(v))
    for i = 1:length(v)
        if v[i] == :var fv[i] = vlink else fv[i] = rholinkpsigmoid end
    end
    fv
end
@inline function varlinkrvec(v)
    fv = Vector{Function}(undef, length(v))
    for i = 1:length(v)
        if v[i] == :var fv[i] = vlinkr else fv[i] = rholinkpsigmoidr end
    end
    fv
end

@inline function varlinkvecapply(v, f)
    rv = similar(v)
    for i = 1:length(v)
        rv[i] = f[i](v[i])
    end
    rv
end
@inline function varlinkvecapply!(v, f)
    for i = 1:length(v)
        v[i] = f[i](v[i])
    end
    v
end

################################################################################
