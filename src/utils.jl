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
    Intersect dataframe.
"""
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
            v2[i] = sort!(v[i][u[i2, i]])
        end
        res[i2] = collect(intersect(Set.(v2)...))
        #res[i2] = intersect(v2...)
    end
    res
end

function intersectsubj(covstr)
    a  = Vector{Vector{Symbol}}(undef, length(covstr.random)+1)
    eq = true
    for i = 1:length(covstr.random)
        a[i] = covstr.random[i].subj
    end
    a[end] = covstr.repeated.subj
    for i = 2:length(a)
        if !(issetequal(a[1], a[i]))
            eq = false
            break
        end
    end
    intersect(a...), eq
end
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

function diffsubj!(a, subj)
    push!(a, subj)
    symdiff(a...)
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
#=
function varlinkvecapply(v, f)
    rv = similar(v)
    for i = 1:length(v)
        rv[i] = f[i](v[i])
    end
    rv
end
=#
function varlinkvecapply!(v, f)
    for i = 1:length(v)
        v[i] = f[i](v[i])
    end
    v
end
varlinkvecapply(v, f) = map.(f,v) #Test
################################################################################
function varlinkvecapply2!(v, p; varlinkf = :exp, rholinkf = :sigm)
    for i = 1:length(v)
        if p[i] == :var
            v[i] = vlink(v[i])
        else
            v[i] = rholinksigmoid(v[i])
        end
    end
    v
end
function varlinkrvecapply2!(v, p; varlinkf = :exp, rholinkf = :sigm)
    for i = 1:length(v)
        if p[i] == :var
            v[i] = vlinkr(v[i])
        else
            v[i] = rholinksigmoidr(v[i])
        end
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
################################################################################



function optim_callback(os)
    false
end
