################################################################################
#=
function makeblocks(subjz)
    blocks = Vector{Vector{UInt32}}(undef, 0)
    for i = 1:size(subjz, 2)
        b = findall(x->!iszero(x), view(subjz, :, i))
        if length(b) > 0 push!(blocks, b) end
    end
    blocks
end

function noncrossmodelmatrix(mx::AbstractArray, my::AbstractArray)
    size(mx, 2) > size(my, 2) ?  (mat = mx' * my; a = mx) : (mat = my' * mx; a = my)
    #mat = mat * mat'
    T = eltype(mat)
    @inbounds for n = 1:size(mat, 2)-1
        fr = findfirst(x->!iszero(x), view(mat, n, :))
        if !isnothing(fr)
            @inbounds for m = fr:size(mat, 1)
                if !iszero(mat[m, n])
                    fc = findfirst(x->!iszero(x), view(mat, m, n+1:size(mat, 2)))
                    if !isnothing(fc)
                        @inbounds for c = n+fc:size(mat, 2)
                            if !iszero(mat[m, c])
                                #view(mat, :, n) .+= view(mat, :, c)
                                A = view(mat, :, n)
                                broadcast!(+, A, A, view(mat, :, c))
                                fill!(view(mat, :, c), zero(T))
                            end
                        end
                    end
                end
            end
        end
    end
    cols = Vector{Int}(undef, 0)
    @inbounds for i = 1:size(mat, 2)
        if !iszero(sum(view(mat,:, i)))
            push!(cols, i)
        end
    end
    res = replace(x -> iszero(x) ?  0 : 1, view(mat, :, cols))
    a * res
end

function diag!(v, m)
    l = checksquare(m)
    l == length(v) || error("Length not equal")
    for i = 1:l
        v[i] = m[i, i]
    end
    v
end

function fillzeroutri!(a::AbstractArray{T}) where T
    s = size(a,1)
    if s == 1 return @inbounds a[1,1] = zero(T) end
    @simd for m = 1:s
        @simd for n = m:s
            @inbounds a[m,n] = zero(T)
        end
    end
    a
end

function logerror!(e, lmm)
    if isa(e, DomainError)
        lmmlog!(lmm, LMMLogMsg(:ERROR, "DomainError ($(e.val), $(e.msg)) during REML calculation."))
    elseif isa(e, BoundsError)
        lmmlog!(lmm, LMMLogMsg(:ERROR, "BoundsError ($(e.a), $(e.i)) during REML calculation."))
    elseif isa(e, ArgumentError)
        lmmlog!(lmm, LMMLogMsg(:ERROR, "ArgumentError ($(e.msg)) during REML calculation."))
    elseif isa(e, LinearAlgebra.SingularException)
        lmmlog!(lmm, LMMLogMsg(:ERROR, "SingularException ($(e.info)) during REML calculation."))
    elseif isa(e, MethodError)
        lmmlog!(lmm, LMMLogMsg(:ERROR, "MethodError ($(e.f), $(e.args), $(e.world)) during REML calculation."))
    else
        lmmlog!(lmm, LMMLogMsg(:ERROR, "Unknown error during REML calculation."))
    end
end

function sweep_β(lmm, data::AbstractLMMDataBlocks, θ::Vector{T}) where T <: Number
    n             = length(lmm.covstr.vcovblock)
    θ₂            = zeros(T, lmm.rankx, lmm.rankx)
    βm            = zeros(T, lmm.rankx)
    β             = Vector{T}(undef, lmm.rankx)
    #---------------------------------------------------------------------------
    akk           = zeros(T, lmm.covstr.maxn + lmm.rankx) #temp for sweep
    Vm            = Matrix{T}(undef, lmm.covstr.maxn + lmm.rankx, lmm.covstr.maxn + lmm.rankx) #!!
    @inbounds @simd for i = 1:n
        q    = length(lmm.covstr.vcovblock[i])
        qswm = q + lmm.rankx
        Vp   = view(Vm, 1:q + lmm.rankx, 1:q + lmm.rankx)
        V    = view(Vm, 1:q, 1:q)
        fillzeroutri!(V)
        Vx   = view(Vm, 1:q, q+1:q+lmm.rankx)
        copyto!(Vx, data.xv[i])
        Vc   = view(Vm, q + 1:qswm, q + 1:qswm)
        fillzeroutri!(Vc)
        vmatrix!(V, θ, lmm, i)
        #-----------------------------------------------------------------------
        sweepb!(fill!(view(akk, 1:qswm), zero(T)), Vp, 1:q)
        subutri!(θ₂, view(Vp, q + 1:qswm, q + 1:qswm))
        mulαtβinc!(βm, view(Vp, 1:q, q + 1:qswm), data.yv[i])
    end
    mul!(β, inv(Symmetric(θ₂)), βm)
    return  β
end

function gmat_zero!(mx, θ::Vector{T}, ::Int, ::CovarianceType) where T
    mx .= zero(T)
    nothing
end

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

function makepmatrix(m::Matrix)
    sm = string.(m)
    lv = maximum(length.(sm), dims = 1)
    for r = 1:size(sm, 1)
        for c = 1:size(sm, 2)
            sm[r,c] = addspace(sm[r,c], lv[c] - length(sm[r,c]))*"   "
        end
    end
end

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

function gmat_blockdiag(θ::Vector{T}, covstr) where T
    vm = Vector{AbstractMatrix{T}}(undef, length(covstr.ves) - 1)
    for i = 1:length(covstr.random)
        vm[i] = gmat(θ[covstr.tr[i]], covstr.q[i], covstr.random[i].covtype, Val{covstr.random[i].covtype.s}()) #covstr.random[i])
    end
    BlockDiagonal(vm)
end

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

function diffsubj!(a, subj)
    push!(a, subj)
    symdiff(a...)
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

#varlinkvecapply(v, f) = map.(f,v) #Test

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

"""
    -2 log Restricted Maximum Likelihood;
"""

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

"""
A * B * A' + C
"""
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

"""
A * B * A'
"""

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
"""
A' * B * A -> +θ
A' * B * C -> +β
"""

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

"""
A' * B * A -> + θ
"""

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

"""
A' * B * A -> θ
"""
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

"""
A * B * A -> θ
"""

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

"""
tr(A * B)
"""

function trmulαβ(A, B)
    c = 0
    @inbounds for n = 1:size(A,1), m = 1:size(B, 1)
        c += A[n,m] * B[m, n]
    end
    c
end

"""
tr(H * A' * B * A)
"""

function trmulhαtβα(H, A, B)
end


"""
(y - X * β)
"""

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


function get_symb(t::Union{ConstantTerm, InterceptTerm, FunctionTerm}; v = Vector{Symbol}(undef, 0))
    v
end
function get_symb(t::Union{Term, CategoricalTerm}; v = Vector{Symbol}(undef, 0))
    push!(v, t.sym)
    v
end
function get_symb(t::InteractionTerm; v = Vector{Symbol}(undef, 0))
    for i in t.terms
        get_symb(i; v = v)
    end
    v
end
function get_symb(t::Tuple{Vararg{AbstractTerm}}; v = Vector{Symbol}(undef, 0))
    for i in t
        get_symb(i; v = v)
    end
    v
end

"""
```math
    \\begin{bmatrix} A * B * A' & X \\\\ X' & 0 \\end{bmatrix}
```
"""
function mulαβαt3(A, B, X)
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
        end
    end
    mx[1:p, p+1:end] = X
    mx[p+1:end, 1:p] = X'
    mx
end

"""
A' * B * A -> + θ
"""
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

"""
A' * B * A -> + θ
"""
function mulαtβαinc!(θ::AbstractMatrix{T}, A::AbstractMatrix, B::AbstractMatrix) where T
    axb  = axes(B, 1)
    sa   = size(A, 2)
    for n ∈ 1:sa
        for m ∈ 1:n
            θmn = zero(T)
            for j ∈ axb
                Ajn = A[j, n]
                for i ∈ axb
                    θmn +=  A[i, m] * B[i, j] * Ajn
                end
            end
            θ[m, n] += θmn
        end
    end
    θ
end

function trmulαβ(A, B)
    c = 0
    @inbounds for n = 1:size(A,1), m = 1:size(B, 1)
        c += A[n,m] * B[m, n]
    end
    c
end

function covmat_grad(f, Z, θ)
    #fx = x -> f(gmat(x[3:5]), rmat(x[1:2], Z), Z)
    #cfg   = ForwardDiff.JacobianConfig(fx, θ)
    m     = ForwardDiff.jacobian(x -> f(gmat(x[3:5]), rmat(x[1:2], Z), Z), θ);
    reshape(m, size(Z, 1), size(Z, 1), length(θ))
end

"""
    2 log Restricted Maximum Likelihood gradient vector
"""

#function reml_grad(yv, Zv, p, Xv, θvec, β)
function reml_grad(lmm, data, θ::Vector{T}, β; syrkblas::Bool = false)  where T
    n     = length(lmm.covstr.vcovblock)

    #G     = gmat(θvec[3:5])
    gvec          = gmatvec(θ, lmm.covstr)

    θ₁    = zeros(length(θvec))
    θ₂    = zeros(length(θvec))
    θ₃    = zeros(length(θvec))

    iV    = Vector{Matrix{T}}(undef, n)

    θ₂m   = zeros(T, lmm.rankx, lmm.rankx)
    H     = zeros(T, lmm.rankx, lmm.rankx)

    for i = 1:n
        q    = length(lmm.covstr.vcovblock[i])
        V    = zeros(q, q)
        vmatrix!(V, gvec, θ, lmm, i)
        iV[i] = inv(V)
        mulαtβαinc!(H, data.xv[i], iV[i])
        #H .+= Xv[i]'*inv(vmat(G, rmat(θvec[1:2], Zv[i]), Zv[i]))*Xv[i]
    end
    iH = inv(H)
    #fx = x -> vmat(gmat(x[3:5]), rmat(x[1:2], Zv[1]), Zv[1])
    #cfg   = ForwardDiff.JacobianConfig(fx, θvec)
    for i = 1:n
        #V   = vmat(G, rmat(θvec[1:2], Zv[i]), Zv[i])
        #iV  = inv(V)
        r   = data.yv[i] .- data.xv[i]*β
        jV  = covmat_grad(vmat, Zv[i], θvec) #!!!
        Aj  = zeros(length(data.yv[i]), length(data.yv[i]))
        for j = 1:length(θvec)

            mulαβαc!(Aj, iV[i], view(jV, :, :, j))
            #Aj      = iV[i] * view(jV, :, :, j) * iV[i]

            θ₁[j]  += trmulαβ(iV[i], view(jV, :, :, j))
            #θ1[j]  += tr(iV[i] * view(jV, :, :, j))

            θ₂[j]  -= tr(iH * data.xv[i]' *Aj * data.xv[i])

            θ₃[j]  -= r' * Aj * r
        end
    end
    return - (θ₁ .+ θ₂ .+ θ₃)
end

"""
    2 log Restricted Maximum Likelihood hessian matrix
"""

function reml_hessian(yv, Zv, p, Xv, θvec, β)
    n     = length(yv)
    G     = gmat(θvec[3:5])
    θ1    = zeros(length(θvec))
    θ2    = zeros(length(θvec))
    θ3    = zeros(length(θvec))
    iV    = nothing
    θ2m   = zeros(p,p)
    H     = zeros(p, p)
    for i = 1:n
        H += Xv[i]'*inv(vmat(rmat(θvec[1:2], Zv[i]), G, Zv[i]))*Xv[i]
    end
    iH = inv(H)
    for i = 1:n
        vmatdvec = x -> vmat(rmat(x[1:2], Zv[i]), gmat(x[3:end]), Zv[i])[:]
        V   = vmat(rmat(θvec[1:2], Zv[i]), G, Zv[i])
        iV  = inv(V)
        r   = yv[i] .- Xv[i]*β
        ∇V  = covmat_grad(covmat, θ, cfg)
        ∇²V = covmat_hessian(covmat, θ)
        #Aij        = iV*∇V[j]*iV
        #Aijk       = -iV * (∇V[k] * iV * ∇V[j] - ∇²V[k,j] + ∇V[j] * iV * ∇V[k]) * iV

        #θ1[j]  += tr(iV * ∇V[j])
        #θ2[j]  -= tr(iH * Xv[i]' * Aij * Xv[i])
        #θ3[j]  -= r' * Aij * r

        #θ1[j,k]  += tr( - Aik' * ∇V[j] + iV * ∇²V[j,k])
        #θ2[j,k]  = - tr( iH * sum(X' * Aik * X) * iH * sum(X' * Aij * X)) - tr(iH * sum(X' * Aijk * X))
        #θ3[j,k]  -= r' * Aijk * r

        #A[j,k] =

        for j = 1:length(θvec)
            for k = 1:length(θvec)

            #θ1[j,k]  +=

            #θ2[j,k]  -=

            #θ3[j,k]  -=
            end
        end

    end
    return - (θ1 .+ θ2 .+ θ3)
end

function reml_grad2(yv, Zv, p, Xv, θvec, β)
    n     = length(yv)
    for i = 1:n
        jV  = covmat_grad(vmat, Zv[i], θvec)
    end
end
function reml_grad3(yv, Zv, p, Xv, θvec, β)
    n     = length(yv)
    covmat_grad2(vmatvec, Zv, θvec)
end

function logdet_(C::Cholesky)
    dd = zero(real(eltype(C)))
    noerror = true
    @inbounds for i in 1:size(C.factors,1)
        v = real(C.factors[i,i])
        if v > 0
            dd += log(v)
        else
            C.factors[i,i] *= -1e-8
            dd += log(real(C.factors[i,i]+4eps()))
            noerror = false
        end
    end
    dd + dd, noerror
end

@noinline function zgz_base_inc!(mx::AbstractArray, θ::AbstractArray{T}, covstr, block, sblock) where T
    if covstr.random[1].covtype.z
        for r = 1:covstr.rn
            G = fill!(Symmetric(Matrix{T}(undef, covstr.q[r], covstr.q[r])), zero(T))
            gmat!(G.data, view(θ, covstr.tr[r]), covstr.random[r].covtype.s)
            zblock    = view(covstr.z, block, covstr.zrndur[r])
            @inbounds for i = 1:length(sblock[r])
                mulαβαtinc!(view(mx, sblock[r][i], sblock[r][i]), view(zblock, sblock[r][i], :), G)
            end
        end
    end
    mx
end

function gmat_switch!(G, θ, covstr, r)
    gmat!(G, view(θ, covstr.tr[r]), covstr.random[r].covtype.s)
    G
end

function tpnum(m, n, s)
    div(m*(2s - 1 - m), 2) - s + n 
end

# use dot(a,b,a) instead

"""
a' * B * a
"""
function mulαtβα(a::AbstractVector, B::AbstractMatrix{T}) where T
    if length(a) != size(B, 2)::Int  || size(B, 2)::Int  != size(B, 1)::Int  error("Dimention error") end
    axbm  = axes(B, 1)
    axbn  = axes(B, 2)
    c = zero(T)
    for i ∈ axbm
        ct = zero(T)
        for j ∈ axbn
            @inbounds  ct += B[i, j] * a[j]
        end
        @inbounds c += ct * a[i]
    end
    c
end

function mulαβαtinc!(θ::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix)
    axb  = axes(B, 1)
    sa   = size(A, 1)
    for m ∈ 1:sa
        for n ∈ m:sa
            for j ∈ axb
                @inbounds for i ∈ axb
                    θ[m, n] +=  A[m, i] * B[i, j] * A[n, j]
                end
            end
        end
    end
    θ
end

function mulαβαtinc!(θ::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, alpha)
    if  !(size(B, 1) == size(B, 2) == size(A, 2)) || !(size(A, 1) == size(θ, 1) == size(θ, 2)) throw(ArgumentError("Wrong dimentions!")) end
    axb  = axes(B, 1)
    sa   = size(A, 1)
    for m ∈ 1:sa
        for n ∈ m:sa
            for j ∈ axb
                for i ∈ axb
                    @inbounds  θ[m, n] +=  A[m, i] * B[i, j] * A[n, j] * alpha
                end
            end
        end
    end
    θ
end

function mulαβαtinc!(θ::AbstractVector, A::AbstractMatrix, B::AbstractMatrix, a::AbstractVector, b::AbstractVector, alpha)
    if !(size(B, 2) == length(a) == length(b)) || size(B, 1) != size(A, 2) || size(A, 1) != length(θ) throw(ArgumentError("Wrong dimentions.")) end
    axb  = axes(B, 1)
    sa   = size(A, 1)
    for m ∈ 1:sa
        for j ∈ axb
            Amj = A[m, j]
            for i ∈ axb
                @inbounds θ[m] +=  Amj * B[j, i] * (a[i] - b[i]) * alpha
            end
        end
    end
    θ
end

function mulθ₃(y, X, β, V::AbstractArray{T}) where T # check for optimization
    q = size(V, 1)
    p = size(X, 2)
    θ = zero(T)

    if q == 1
        cs = zero(T)
        @inbounds  for m in 1:p
            cs += X[1, m] * β[m]
        end
        return -V[1, 1] * (y[1] - cs)^2
    end

    c = zeros(T, q)
    for n = 1:q
        for m = 1:p
            c[n] += X[n, m] * β[m]
        end
    end

    @simd for n = 1:q-1
        ycn = y[n] - c[n]
        @simd for m = n+1:q
            @inbounds θ -= V[n, m] * ycn * (y[m] - c[m]) * 2
        end
    end
    @inbounds  for m = 1:q
        θ -= V[m, m] * (y[m] - c[m]) ^ 2
    end

    return θ
end

function addspace(s::String, n::Int; first = false)::String
    if n > 0
        for i = 1:n
            if first s = Char(' ') * s else s = s * Char(' ') end
        end
    end
    return s
end

function printmatrix(io::IO, m::Matrix; header = false)
    sm = string.(m)
    lv = maximum(length.(sm), dims = 1)
    for r = 1:size(sm, 1)
        for c = 1:size(sm, 2)
            sm[r,c] = addspace(sm[r,c], lv[c] - length(sm[r,c]))*"   "
        end
    end
    i = 1
    if header
        line = "⎯" ^ sum(length.(sm[1,:]))
        println(io, line)
        for c = 1:size(sm, 2)
            print(io, sm[1,c])
        end
        print(io, "\n")
        println(io, line)
        i = 2
    end
    for r = i:size(sm, 1)
        for c = 1:size(sm, 2)
            print(io, sm[r,c])
        end
        print(io, "\n")
    end
end

function vmatrix!(V, G, θ, lmm::LMM, i::Int)
    zgz_base_inc!(V, G, θ, lmm.covstr, lmm.covstr.vcovblock[i], lmm.covstr.sblock[i])
    rmat_base_inc!(V, θ[lmm.covstr.tr[end]], lmm.covstr, lmm.covstr.vcovblock[i], lmm.covstr.sblock[i])
end

function grad_vmatrix(θ::AbstractVector{T}, lmm::LMM, i::Int)
    V    = zeros(T, length(lmm.covstr.vcovblock[i]), length(lmm.covstr.vcovblock[i]))
    gvec = gmatvec(θ, lmm.covstr)
    vmatrix!(V, gvec, θ, lmm, i)
    Symmetric(V)
end

# logdet with check
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

function indsdict!(d::Dict, cdata::Union{Tuple, NamedTuple, AbstractVector{AbstractVector}})
    @inbounds for (i, element) in enumerate(zip(cdata...))
        ind = ht_keyindex(d, element)
        if ind > 0
            push!(d.vals[ind], i)
        else
            v = Vector{Int}(undef, 1)
            v[1] = i
            d[element] = v
        end
    end
    d
end
function indsdict!(d::Dict, cdata::AbstractVector)
    for i = 1:length(cdata)
        ind = ht_keyindex(d, cdata[i])
        if ind > 0
            push!(d.vals[ind], i)
        else
            v = Vector{Int}(undef, 1)
            v[1] = i
            d[cdata[i]] = v
        end
    end
    d
end


function fill_coding_dict!(t::T, d::Dict, data) where T <: CategoricalTerm
    if typeof(data[!, t.sym])  <: CategoricalArray
        d[t.sym] = StatsModels.FullDummyCoding()
    end
end

function fulldummycodingdict(t::InteractionTerm)
    d = Dict{Symbol, AbstractContrasts}()
    for i in t.terms
        d[i.sym] = StatsModels.FullDummyCoding()
    end
    d
end
function fulldummycodingdict(t::T) where T <: Union{CategoricalTerm, Term}
    d = Dict{Symbol, AbstractContrasts}()
    d[t.sym] = StatsModels.FullDummyCoding()
    d
end
function fulldummycodingdict(t::T) where T <: Union{ConstantTerm, InterceptTerm}
    d = Dict{Symbol, AbstractContrasts}()
    d
end
function fulldummycodingdict(t::Tuple{Vararg{AbstractTerm}})
    d = Dict{Symbol, AbstractContrasts}()
    for i in t
        merge!(d, fulldummycodingdict(i))
    end
    d
end

=#