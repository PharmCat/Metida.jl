#reml.jl
function subutri!(a, b)
    s = size(a,1)
    if s == 1 return @inbounds a[1,1] -= b[1,1] end
    @simd for m = 1:s
        @simd for n = m:s
            @inbounds a[m,n] -= b[m,n]
        end
    end
    a
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
################################################################################
#                     REML without provided β
################################################################################
function reml_sweep_β(lmm, θ::Vector{T}) where T <: Number
    data = LMMDataViews(lmm)
    reml_sweep_β(lmm, data, θ)
end
function reml_sweep_β(lmm, data::AbstractLMMDataBlocks, θ::Vector{T}) where T <: Number
    set_zero_subnormals(true)
    n             = length(lmm.covstr.vcovblock)
    N             = length(lmm.data.yv)
    c             = (N - lmm.rankx)*log(2π)
    #---------------------------------------------------------------------------
    V⁻¹           = Vector{AbstractArray{T}}(undef, n)
    θ₁            = zero(T)
    θ₂            = zeros(T, lmm.rankx, lmm.rankx)
    θ₃            = zero(T)
    βm            = zeros(T, lmm.rankx)
    β             = Vector{T}(undef, lmm.rankx)
    #---------------------------------------------------------------------------
    local logdetθ₂
    local θs₂
    akk = Vector{T}(undef, lmm.covstr.maxn + lmm.rankx) #temp for sweep
    try
        @inbounds @simd for i = 1:n #@fastmath
            q    = length(lmm.covstr.vcovblock[i])
            qswm = q + lmm.rankx
            Vp   = Matrix{T}(undef, qswm, qswm)
            V    = view(Vp, 1:q, 1:q)
            Vx   = view(Vp, 1:q, q+1:qswm)
            Vc   = view(Vp, q+1:qswm, q+1:qswm)
            fillzeroutri!(V)
            copyto!(Vx, data.xv[i])
            fillzeroutri!(Vc)
            vmatrix!(V, θ, lmm, i)
            #-----------------------------------------------------------------------
            swr  = sweepb!(fill!(view(akk, 1:qswm), zero(T)), Vp, 1:q; logdet = true)
            θ₁  += swr[2]
            V⁻¹[i] = V
            #-----------------------------------------------------------------------
            subutri!(θ₂, Vc)
            #θ₂ -= UpperTriangular(view(Vp, q + 1:qswm, q + 1:qswm))
            mulαtβinc!(βm, Vx, data.yv[i])
            #-----------------------------------------------------------------------
        end
        θs₂ = Symmetric(θ₂)
        mul!(β, inv(θs₂), βm)
        @inbounds @simd for i = 1:n
            θ₃ += mulθ₃(data.yv[i], data.xv[i], β, V⁻¹[i])
        end
        logdetθ₂ = logdet(θs₂)
    catch e
        logerror!(e, lmm)
        return (Inf, nothing, nothing, nothing, false)
    end
    return   θ₁ + logdetθ₂ + θ₃ + c, β, θs₂, θ₃, true #REML, β, iC, θ₃, errors
end
################################################################################
#                     REML with provided β
################################################################################
function reml_sweep_β(lmm, θ::Vector{T}, β::Vector) where T <: Number
    data = LMMDataViews(lmm)
    reml_sweep_β(lmm, data, θ, β)
end
function reml_sweep_β(lmm, data::AbstractLMMDataBlocks, θ::Vector{T}, β::Vector) where T <: Number
    n             = length(lmm.covstr.vcovblock)
    N             = length(lmm.data.yv)
    c             = (N - lmm.rankx)*log(2π)
    #---------------------------------------------------------------------------
    θ₁            = zero(T)
    θ₂            = zeros(T, lmm.rankx, lmm.rankx)
    θ₃            = zero(T)
    #---------------------------------------------------------------------------
    logdetθ₂      = zero(T)
    akk           = zeros(T, lmm.covstr.maxn + lmm.rankx) #temp for sweep
    Vm            = Matrix{T}(undef, lmm.covstr.maxn + lmm.rankx, lmm.covstr.maxn + lmm.rankx) #!!
    local θs₂
    try
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
            swr  = sweepb!(fill!(view(akk, 1:qswm), zero(T)), Vp, 1:q; logdet = true)
            θ₁  += swr[2]
            #-----------------------------------------------------------------------
            subutri!(θ₂, view(Vp, q + 1:qswm, q + 1:qswm))
            θ₃  += mulθ₃(data.yv[i], data.xv[i], β, V)
        end
        θs₂ = Symmetric(θ₂)
        logdetθ₂ = logdet(θs₂)
    catch e
        logerror!(e, lmm)
        return (Inf, nothing, 1e100, false)
    end
    return   θ₁ + logdetθ₂ + θ₃ + c, θs₂, θ₃, true #REML, iC, θ₃
end
################################################################################
#                     REML AI-like / scoring part
################################################################################
function sweep_ai(lmm, data::AbstractLMMDataBlocks, θ::Vector{T}, β::Vector) where T <: Number
    n             = length(lmm.covstr.vcovblock)
    θ₃            = zero(T)
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
        sweepb!(fill!(view(akk, 1:qswm), zero(T)), Vp, 1:q)
        θ₃  += mulθ₃(data.yv[i], data.xv[i], β, V)
    end
    return  θ₃
end
function sweep_score(lmm, data::AbstractLMMDataBlocks, θ::Vector{T}, β::Vector) where T <: Number
    n             = length(lmm.covstr.vcovblock)
    θ₁            = zero(T)
    θ₃            = zero(T)
    #---------------------------------------------------------------------------
    akk           = zeros(T, lmm.covstr.maxn + lmm.rankx) #temp for sweep
    Vm            = Matrix{T}(undef, lmm.covstr.maxn + lmm.rankx, lmm.covstr.maxn + lmm.rankx) #!!
    try
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
        swr  = sweepb!(fill!(view(akk, 1:qswm), zero(T)), Vp, 1:q; logdet = true)
        θ₁  += swr[2]
        θ₃  += mulθ₃(data.yv[i], data.xv[i], β, V)
    end
    catch
        logerror!(e, lmm)
        return Inf
    end
    return   -θ₁ + θ₃
end
################################################################################
#                     β calculation
################################################################################
#=
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
=#
################################################################################
#                     variance-covariance matrix of β
################################################################################
function sweep_β_cov(lmm, θ::Vector{T}, β::Vector) where T <: Number
    n             = length(lmm.covstr.vcovblock)
    θ₂            = zeros(T, lmm.rankx, lmm.rankx)
    akk           = zeros(T, lmm.covstr.maxn + lmm.rankx) #temp for sweep
    Vm            = Matrix{T}(undef, lmm.covstr.maxn + lmm.rankx, lmm.covstr.maxn + lmm.rankx) #!!
    @inbounds @simd for i = 1:n
        q    = length(lmm.covstr.vcovblock[i])
        qswm = q + lmm.rankx
        Vp   = view(Vm, 1:q + lmm.rankx, 1:q + lmm.rankx)
        V    = view(Vm, 1:q, 1:q)
        fillzeroutri!(V)
        Vx   = view(Vm, 1:q, q+1:q+lmm.rankx)
        copyto!(Vx,  view(lmm.data.xv,  lmm.covstr.vcovblock[i],:))
        Vc   = view(Vm, q + 1:qswm, q + 1:qswm)
        fillzeroutri!(Vc)
        vmatrix!(V, θ, lmm, i)
        #-----------------------------------------------------------------------
        sweepb!(fill!(view(akk, 1:qswm), zero(T)), Vp, 1:q)
        subutri!(θ₂, view(Vp, q + 1:qswm, q + 1:qswm))
    end
    return Symmetric(θ₂)
end
