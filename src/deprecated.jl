#=
function gmat_zero!(mx, θ::Vector{T}, ::Int, ::CovarianceType) where T
    mx .= zero(T)
    nothing
end
=#
#=
function lcontrast(lmm::LMM, i::Int)
    n = nterms(lmm.mf)
    if i > n || n < 1 error("Factor number out of range 1-$(n)") end
    inds = findall(x -> x==i, lmm.mm.assign)
    mx = zeros(length(inds), size(lmm.mm.m, 2))
    for i = 1:length(inds)
        mx[i, inds[i]] = 1
    end
    mx
end
=#
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
#=
function makepmatrix(m::Matrix)
    sm = string.(m)
    lv = maximum(length.(sm), dims = 1)
    for r = 1:size(sm, 1)
        for c = 1:size(sm, 2)
            sm[r,c] = addspace(sm[r,c], lv[c] - length(sm[r,c]))*"   "
        end
    end
end
=#
#=
function gmat(θ::Vector{T}, zn::Int, ::CovarianceType, ::Val{:SI}) where T
    Matrix{T}(I(zn)*(θ[1] ^ 2))
    #I(zn)*(θ[1] ^ 2)
end
function gmat(θ::Vector{T}, ::Int, ::CovarianceType, ::Val{:VC}) where T
    Matrix{T}(Diagonal(θ .^ 2))
    #Diagonal(θ .^ 2)
end
function gmat(θ::Vector{T}, zn::Int, ::CovarianceType, ::Val{:AR}) where T
    mx  = Matrix{T}(undef, zn, zn)
    mx .= θ[1] ^ 2
    if zn > 1
        for m = 1:zn - 1
            for n = m + 1:zn
                @inbounds mx[m, n] = mx[m, m] * θ[2] ^ (n - m)
            end
        end
    end
    Matrix{T}(Symmetric(mx))
    #Symmetric(mx)
end
function gmat(θ::Vector{T}, zn::Int, ::CovarianceType, ::Val{:ARH}) where T
    mx  = Matrix{T}(undef, zn, zn)
    for m = 1:zn
        @inbounds mx[m, m] = θ[m]
    end
    if zn > 1
        for m = 1:zn - 1
            for n = m + 1:zn
                @inbounds mx[m, n] = mx[m, m] * mx[n, n] * θ[end] ^ (n - m)
            end
        end
    end
    for m = 1:zn
        @inbounds mx[m, m] = mx[m, m] * mx[m, m]
    end
    Matrix{T}(Symmetric(mx))
    #Symmetric(mx)
end
function gmat(θ::Vector{T}, zn::Int, ::CovarianceType, ::Val{:CSH}) where T
    mx = Matrix{T}(undef, zn, zn)
    for m = 1:zn
        @inbounds mx[m, m] = θ[m]
    end
    if zn > 1
        for m = 1:zn - 1
            for n = m + 1:zn
                @inbounds mx[m, n] = mx[m, m] * mx[n, n] * θ[end]
            end
        end
    end
    for m = 1:zn
        @inbounds mx[m, m] = mx[m, m] * mx[m, m]
    end
    Matrix{T}(Symmetric(mx))
    #Symmetric(mx)
end
=#
#=
function gmat_blockdiag(θ::Vector{T}, covstr) where T
    vm = Vector{AbstractMatrix{T}}(undef, length(covstr.ves) - 1)
    for i = 1:length(covstr.random)
        vm[i] = gmat(θ[covstr.tr[i]], covstr.q[i], covstr.random[i].covtype, Val{covstr.random[i].covtype.s}()) #covstr.random[i])
    end
    BlockDiagonal(vm)
end
=#
#=
@inline function rmat(θ::Vector{T}, rz, rn, ct, ::Val{:SI}) where T
    I(rn) * (θ[1] ^ 2)
end
@inline function rmat(θ::Vector{T}, rz, rn, ct, ::Val{:VC}) where T
    Diagonal(rz * (θ .^ 2))
end
@inline function rmat(θ::Vector{T}, rz, rn, ct, ::Val{:AR}) where T
    mx  = Matrix{T}(undef, rn, rn)
    mx .= θ[1] ^ 2
    if rn > 1
        for m = 1:rn - 1
            for n = m + 1:rn
                @inbounds mx[m, n] = mx[m, m] * θ[2] ^ (n - m)
            end
        end
    end
    Symmetric(mx)
end
@inline function rmat(θ::Vector{T}, rz, rn, ct, ::Val{:ARH}) where T
    mx   = Matrix(Diagonal(rz * (θ[1:end-1])))
    if rn > 1
        for m = 1:rn - 1
            for n = m + 1:rn
                @inbounds mx[m, n] = mx[m, m] * mx[n, n] * θ[end] ^ (n - m)
            end
        end
    end
    for m = 1:rn
        @inbounds mx[m, m] = mx[m, m] * mx[m, m]
    end
    Symmetric(mx)
end
@inline function rmat(θ::Vector{T}, rz, rn,  ct, ::Val{:CSH}) where T #???
    mx   = Matrix(Diagonal(rz * (θ[1:end-1])))
    if rn > 1
        for m = 1:rn - 1
            for n = m + 1:rn
                @inbounds mx[m, n] = mx[m, m] * mx[n, n] * θ[end]
            end
        end
    end
    for m = 1:rn
        @inbounds mx[m, m] = mx[m, m] * mx[m, m]
    end
    Symmetric(mx)
end
=#
#=
function get_z_matrix(data, covstr::CovStructure{Vector{VarEffect}})
    rschema = apply_schema(covstr.random[1].model, schema(data, covstr.random[1].coding))
    Z       = modelcols(rschema, data)
    if length(covstr.random) > 1
        for i = 1:length(covstr.random)
            rschema = apply_schema(covstr.random[i].model, schema(data, covstr.random[i].coding))
            Z       = hcat(modelcols(rschema, data))
        end
    end
    Z
end

function get_z_matrix(data, covstr::CovStructure)
    rschema = apply_schema(covstr.random.model, schema(data, covstr.random.coding))
    Z       = modelcols(rschema, data)
end

function get_term_vec(covstr::CovStructure)
    covstr.random.model
end
=#
#=
"""

"""
#cfg = ForwardDiff.JacobianConfig(covmat, [1.0,2.0,3.0]);
#function covmat_grad(f, θ, cfg)
#    ForwardDiff.jacobian(f, θ, cfg);
#end
function covmat_grad(f, Z, θ)
    #fx = x -> f(gmat(x[3:5]), rmat(x[1:2], Z), Z)
    #cfg   = ForwardDiff.JacobianConfig(fx, θ)
    m     = ForwardDiff.jacobian(x -> f(gmat(x[3:5]), rmat(x[1:2], Z), Z), θ);
    reshape(m, size(Z, 1), size(Z, 1), length(θ))
end
function covmat_hessian(f, θ)
    ForwardDiff.jacobian(x -> ForwardDiff.jacobian(f, x), θ)
end
=#
################################################################################
#=
function covmat_grad2(f, Z, θ)
    #fx = x -> f(gmat(x[3:5]), rmat(x[1:2], Z), Z)
    #cfg   = ForwardDiff.JacobianConfig(fx, θ)

    m     = ForwardDiff.jacobian(x -> f(gmat(x[3:5]), Z, x), θ);
end

function fgh!(F, G, H, x; remlβcalc::Function, remlcalc::Function)

  reml, β, c = remlβcalc(x)
  remlcalcd = x -> remlcalc(β, x)
  if G != nothing
      G .= ForwardDiff.gradient(remlcalcd, x)
  end
  if H != nothing
      H .= ForwardDiff.hessian(remlcalcd, x)
  end
  if F != nothing
    return reml
  end
  nothing
end
=#

"""
    Make X, Z matrices and vector y for each subject;
"""
#=
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
=#


#=
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
=#
#=
function diffsubj!(a, subj)
    push!(a, subj)
    symdiff(a...)
end
=#
#=
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
=#
#varlinkvecapply(v, f) = map.(f,v) #Test

#=
function varlinkrvecapply2(v, p; varlinkf = :exp, rholinkf = :sigm)
    s = similar(v)
    for i = 1:length(v)
        if p[i] == :var
            s[i] = vlinkr(v[i])
        else
            s[i] = rholinksigmoidr(v[i])
        end
    end
    s
end
=#
################################################################################
#=
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
=#

#=
function rmat_basep_z!(mx, θ::AbstractVector{T}, zrv, covstr) where T
    for i = 1:length(covstr.block[end])
        if covstr.repeated.covtype.s == :SI
            rmatp_si!(view(mx, covstr.block[end][i], covstr.block[end][i]), θ, view(covstr.rz, covstr.block[end][i], :), covstr.repeated.covtype)
        elseif covstr.repeated.covtype.s == :DIAG
            rmatp_diag!(view(mx, covstr.block[end][i], covstr.block[end][i]), θ, view(covstr.rz, covstr.block[end][i], :), covstr.repeated.covtype)
        elseif covstr.repeated.covtype.s == :AR
            rmatp_ar!(view(mx, covstr.block[end][i], covstr.block[end][i]), θ, view(covstr.rz, covstr.block[end][i], :), covstr.repeated.covtype)
        elseif covstr.repeated.covtype.s == :ARH
            rmatp_arh!(view(mx, covstr.block[end][i], covstr.block[end][i]), θ, view(covstr.rz, covstr.block[end][i], :), covstr.repeated.covtype)
        elseif covstr.repeated.covtype.s == :CSH
            rmatp_csh!(view(mx, covstr.block[end][i], covstr.block[end][i]), θ, view(covstr.rz, covstr.block[end][i], :), covstr.repeated.covtype)
        elseif covstr.repeated.covtype.s == :CS
            rmatp_cs!(view(mx, covstr.block[end][i], covstr.block[end][i]), θ, view(covstr.rz, covstr.block[end][i], :), covstr.repeated.covtype)
        else
            error("Unknown covariance structure!")
        end
    end
    mx
end
=#

"""
    -2 log Restricted Maximum Likelihood;
"""
#=
function reml_sweep(lmm, β, θ::Vector{T})::T where T <: Number
    n  = length(lmm.data.yv)
    N  = sum(length.(lmm.data.yv))
    G  = gmat_base(θ, lmm.covstr)
    c  = (N - lmm.rankx)*log(2π)
    θ₁ = zero(eltype(θ))
    θ₂ = zero(eltype(θ))
    θ₃ = zero(eltype(θ))
    θ2m  = zeros(eltype(θ), lmm.rankx, lmm.rankx)
    @inbounds for i = 1:n
        q   = length(lmm.data.yv[i])
        Vp  = mulαβαt3(lmm.data.zv[i], G, lmm.data.xv[i])
        V   = view(Vp, 1:q, 1:q)
        rmat_basep!(V, θ[lmm.covstr.tr[end]], lmm.data.zrv[i], lmm.covstr)

        θ₁  += logdet(V)
        sweep!(Vp, 1:q)
        iV  = Symmetric(utriaply!(x -> -x, Vp[1:q, 1:q]))
        mulαtβαinc!(θ2m, lmm.data.xv[i], iV)
        θ₃  += -Vp[end, end]
    end
    θ₂       = logdet(θ2m)
    return   -(θ₁ + θ₂ + θ₃ + c)
end
=#

#=
function gfunc!(g, x, f)
    chunk  = ForwardDiff.Chunk{1}()
    gcfg   = ForwardDiff.GradientConfig(f, x, chunk)
    ForwardDiff.gradient!(g, f, x, gcfg)
end
function hfunc!(h, x, f)
    chunk  = ForwardDiff.Chunk{1}()
    hcfg   = ForwardDiff.HessianConfig(f, x, chunk)
    ForwardDiff.hessian!(h, f, x, hcfg)
end
=#

"""
A * B * A' + C
"""
#=
function mulαβαtc(A, B, C)
    q  = size(B, 1)
    p  = size(A, 1)
    c  = zeros(eltype(B), q)
    mx = zeros(eltype(B), p, p)
    for i = 1:p
        fill!(c, zero(eltype(c)))
        @simd for n = 1:q
            @simd for m = 1:q
                @inbounds c[n] +=  A[i, m] * B[n, m]
            end
        end
        @simd for n = i:p
            @simd for m = 1:q
                 @inbounds mx[i, n] += A[n, m] * c[m]
            end
            @inbounds mx[i, n] += C[i, n]
        end
    end
    Symmetric(mx)
end
=#
"""
A * B * A'
"""
#=
function mulαβαt(A, B)
    q  = size(B, 1)
    p  = size(A, 1)
    c  = zeros(eltype(B), q)
    mx = zeros(eltype(B), p, p)
    for i = 1:p
        fill!(c, zero(eltype(c)))
        @simd for n = 1:q
            @simd for m = 1:q
                @inbounds c[n] +=  A[i, m] * B[n, m]
            end
        end
        @simd for n = i:p
            @simd for m = 1:q
                 @inbounds mx[i, n] += A[n, m] * c[m]
            end
        end
    end
    Symmetric(mx)
end
function mulαβαtc2(A, B, C, r::Vector)
    q  = size(B, 1)
    p  = size(A, 1)
    c  = zeros(eltype(B), q)
    mx = zeros(eltype(B), p + 1, p + 1)
    for i = 1:p
        fill!(c, zero(eltype(c)))
        @simd for n = 1:q
            @simd for m = 1:q
                @inbounds c[n] +=  A[i, m] * B[n, m]
            end
        end
        @simd for n = 1:p
            @simd for m = 1:q
                 @inbounds mx[i, n] += A[n, m] * c[m]
            end
            @inbounds mx[i, n] += C[i, n]
        end
        mx[end, i] = r[i]
        mx[i, end] = r[i]
    end
    mx
end
function mulαβαtc3(A, B, C, X)
    q  = size(B, 1)
    p  = size(A, 1)
    c  = zeros(eltype(B), q)
    mx = zeros(eltype(B), p + size(X, 2), p + size(X, 2))
    for i = 1:p
        fill!(c, zero(eltype(c)))
        @simd for n = 1:q
            @simd for m = 1:q
                @inbounds c[n] +=  A[i, m] * B[n, m]
            end
        end
        @simd for n = 1:p
            @simd for m = 1:q
                 @inbounds mx[i, n] += A[n, m] * c[m]
            end
            @inbounds mx[i, n] += C[i, n]
        end
    end
    mx[1:p, p+1:end] = X
    mx[p+1:end, 1:p] = X'
    mx
end
=#

"""
A' * B * A -> +θ
A' * B * C -> +β
"""
#=
function mulθβinc!(θ, β, A::AbstractMatrix, B::AbstractMatrix, C::AbstractVector)
    q = size(B, 1)
    p = size(A, 2)
    c = Vector{eltype(θ)}(undef, q)
    for i = 1:p
        fill!(c, zero(eltype(θ)))
        for n = 1:q
            for m = 1:q
                @inbounds c[n] += B[m, n] * A[m, i]
            end
            @inbounds β[i] += c[n] * C[n]
        end
        for n = 1:p
            for m = 1:q
                @inbounds θ[i, n] += A[m, n] * c[m]
            end
        end
    end
    θ, β
end
=#
#-------------------------------------------------------------------------------
"""
A' * B * A -> + θ
"""
#=
function mulαtβαinc!(θ, A, B)
    q = size(B, 1)
    p = size(A, 2)
    c = zeros(eltype(B), q)
    for i = 1:p
        fill!(c, zero(eltype(c)))
        @inbounds for n = 1:q, m = 1:q
            c[n] += B[m, n] * A[m, i]
        end
        @inbounds for n = 1:p, m = 1:q
            θ[i, n] += A[m, n] * c[m]
        end
    end
end
=#

"""
A' * B * A -> θ
"""
#=
function mulαtβα!(θ, A, B)
    q = size(B, 1)
    p = size(A, 2)
    c = zeros(eltype(B), q)
    fill!(θ, zero(eltype(θ)))
    for i = 1:p
        fill!(c, zero(eltype(c)))
        @inbounds for n = 1:q, m = 1:q
            c[n] += B[m, n] * A[m, i]
        end
        @inbounds for n = 1:p, m = 1:q
            θ[i, n] += A[m, n] * c[m]
        end
    end
    θ
end
=#
"""
A * B * A -> θ
"""
#=
function mulαβα!(θ, A, B)
    q = size(B, 1)
    p = size(A, 2)
    c = zeros(eltype(B), q)
    fill!(θ, zero(eltype(θ)))
    for i = 1:p
        fill!(c, zero(eltype(c)))
        @inbounds for n = 1:q, m = 1:q
            c[n] += B[m, n] * A[i, m]
        end
        @inbounds for n = 1:p, m = 1:q
            θ[i, n] += A[m, n] * c[m]
        end
    end
    θ
end
=#
"""
tr(A * B)
"""
#=
function trmulαβ(A, B)
    c = 0
    @inbounds for n = 1:size(A,1), m = 1:size(B, 1)
        c += A[n,m] * B[m, n]
    end
    c
end
=#
"""
tr(H * A' * B * A)
"""
#=
function trmulhαtβα(H, A, B)
end
=#

"""
(y - X * β)
"""
#=
function mulr!(v::AbstractVector, y::AbstractVector, X::AbstractMatrix, β::AbstractVector)
    fill!(v, zero(eltype(v)))
    q = length(y)
    p = size(X, 2)
    @simd for n = 1:q
        @simd for m = 1:p
            @inbounds v[n] += X[n, m] * β[m]
        end
        v[n] = y[n] - v[n]
    end
    return v
end
function mulr(y::AbstractVector, X::AbstractMatrix, β::AbstractVector)
    v = zeros(eltype(β), length(y))
    q = length(y)
    p = size(X, 2)
    @simd for n = 1:q
        @simd for m = 1:p
            @inbounds v[n] += X[n, m] * β[m]
        end
        v[n] = y[n] - v[n]
    end
    return v
end
=#
