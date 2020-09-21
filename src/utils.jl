"""
    Make X, Z matrices and vector y for each subject;
"""
function subjblocks(df, sbj::Symbol, x::Matrix{T1}, z::Matrix{T2}, y::Vector{T3}, rz::T4) where {T1, T2, T3, T4 <: AbstractMatrix}
    u = unique(df[!, sbj])
    v = findall(x -> x == u[1], df[!, sbj])
    xv  = view(x, v, :)
    zv  = view(z, v, :)
    yv  = view(y, v)
    rzv = view(rz, v, :)
    xa  = Vector{typeof(xv)}(undef, length(u))
    za  = Vector{typeof(zv)}(undef, length(u))
    ya  = Vector{typeof(yv)}(undef, length(u))
    rza = Vector{typeof(rzv)}(undef, length(u))
    xa[1] = xv
    za[1] = zv
    ya[1] = yv
    rza[1] = rzv
    #=
    xa  = Vector{SubArray{T1, 2, Matrix{T1}, Tuple{Array{Int64,1}, Base.Slice{Base.OneTo{Int64}}},false}}(undef, length(u))
    za  = Vector{SubArray{T2, 2, Matrix{T2}, Tuple{Array{Int64,1}, Base.Slice{Base.OneTo{Int64}}},false}}(undef, length(u))
    ya  = Vector{SubArray{T3, 1, Vector{T3}, Tuple{Array{Int64,1}},false}}(undef, length(u))
    rza = Vector{SubArray{T4, 2, Matrix{T4}, Tuple{Array{Int64,1}, Base.Slice{Base.OneTo{Int64}}},false}}(undef, length(u))
    =#
    if length(u) > 1
        @inbounds @simd for i = 2:length(u)
            v = findall(x->x==u[i], df[!, sbj])
            xa[i] = view(x, v, :)
            za[i] = view(z, v, :)
            ya[i] = view(y, v)
            rza[i] = view(rz, v, :)
        end
    end
    return xa, za, rza, ya
end
function subjblocks(df, sbj::Symbol, x::Matrix{T1}, z::Matrix{T2}, y::Vector{T3}, rz::Nothing) where {T1, T2, T3}
    u = unique(df[!, sbj])
    v = findall(x -> x == u[1], df[!, sbj])
    xv  = view(x, v, :)
    zv  = view(z, v, :)
    yv  = view(y, v)
    xa  = Vector{typeof(xv)}(undef, length(u))
    za  = Vector{typeof(zv)}(undef, length(u))
    ya  = Vector{typeof(yv)}(undef, length(u))
    rza = Vector{Int}(undef, length(u))
    xa[1] = xv
    za[1] = zv
    ya[1] = yv
    rza[1] = length(ya[1])
    #xa  = Vector{SubArray{T1, 2, Matrix{T1}, Tuple{Array{Int64,1}, Base.Slice{Base.OneTo{Int64}}},false}}(undef, length(u))
    #za  = Vector{SubArray{T2, 2, Matrix{T2}, Tuple{Array{Int64,1}, Base.Slice{Base.OneTo{Int64}}},false}}(undef, length(u))
    #ya  = Vector{SubArray{T3, 1, Vector{T3}, Tuple{Array{Int64,1}},false}}(undef, length(u))
    #rza = Vector{Int}(undef, length(u))
    if length(u) > 1
        @inbounds @simd for i = 2:length(u)
            v = findall(x->x==u[i], df[!, sbj])
            xa[i]  = view(x, v, :)
            za[i]  = view(z, v, :)
            ya[i]  = view(y, v)
            rza[i] = length(ya[i])
        end
    end
    return xa, za, rza, ya
end

"""
Variance estimate via OLS and QR decomposition.
"""
function initvar(y::Vector, X::Matrix{T}) where T
    qrx  = qr(X)
    β    = inv(qrx.R) * qrx.Q' * y
    r    = y - X * b
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

function rholinksigmoid(ρ::T, m) where T <: Real
    return ρ/sqrt(1.0 + ρ^2)
end
function rholinksigmoidr(ρ::T, m) where T <: Real
    return sign(ρ)*sqrt(ρ^2/(1.0 - ρ^2))
end

function rholinksigmoid2(ρ::T, m) where T <: Real
    return atan(ρ)/pi*2.0
end
function rholinksigmoid2r(ρ::T, m) where T <: Real
    return tan(ρ*pi/2.0)
end
