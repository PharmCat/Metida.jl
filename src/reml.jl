#reml.jl

function logdetv(V)
    try
        return logdet(cholesky(V))
    catch
        return logdet(V)
    end
end
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
################################################################################
#                     REML without provided β
################################################################################
function reml_sweep_β(lmm, θ::Vector{T}) where T <: Number
    data = LMMDataViews(lmm)
    reml_sweep_β(lmm, data, θ)
end
function reml_sweep_β(lmm, data::AbstractLMMDataBlocks, θ::Vector{T}) where T <: Number
    noerrors      = true
    n             = length(lmm.covstr.vcovblock)
    N             = length(lmm.data.yv)
    c             = (N - lmm.rankx)*log(2π)
    #---------------------------------------------------------------------------
    V⁻¹           = Vector{AbstractArray{T}}(undef, n)
    # Vector log determinant of V matrix
    θ₁            = zero(T)
    θ₂            = zeros(T, lmm.rankx, lmm.rankx)
    θ₃            = zero(T)
    βm            = zeros(T, lmm.rankx)
    β             = Vector{T}(undef, lmm.rankx)
    #---------------------------------------------------------------------------
    #q             = zero(Int)
    #qswm          = zero(Int)
    #Vp            = Matrix{T}(undef, 0, 0)
    #logdetθ₂      = zero(T)
    local logdetθ₂::T
    akk           = zeros(T, lmm.covstr.maxn + lmm.rankx) #temp for sweep

    @inbounds for i = 1:n
        q    = length(lmm.covstr.vcovblock[i])
        qswm = q + lmm.rankx
        Vp  = zeros(T, qswm, qswm)
        V   = view(Vp, 1:q, 1:q)
        Vx   = view(Vp, 1:q, q+1:qswm)
        Vx  .= data.xv[i]
        vmatrix!(V, θ, lmm, i)
        #-----------------------------------------------------------------------
        try
            θ₁  += logdetv(Symmetric(V))
        catch
            noerrors = false
            lmmlog!(lmm, LMMLogMsg(:ERROR, "θ₁ not estimated during REML calculation, V isn't positive definite or |V| less zero."))
            return (1e100, nothing, nothing, 1e100, noerrors)
        end
        sweepb!(fill!(view(akk, 1:qswm), zero(T)), Vp, 1:q)
        #sweep!(Vp, 1:q)
        V⁻¹[i] = V
        #-----------------------------------------------------------------------
        #θ₂ .-= view(Vp, q + 1:qswm, q + 1:qswm)
        subutri!(θ₂, view(Vp, q + 1:qswm, q + 1:qswm))
        #θ₂ -= UpperTriangular(view(Vp, q + 1:qswm, q + 1:qswm))
        mulαtβinc!(βm, view(Vp, 1:q, q + 1:qswm), data.yv[i])
        #-----------------------------------------------------------------------
    end
    θs₂ = Symmetric(θ₂)
    mul!(β, inv(θs₂), βm)
    @simd for i = 1:n
        @inbounds θ₃  += mulθ₃(data.yv[i], data.xv[i], β, V⁻¹[i])
    end
    try
        logdetθ₂ = logdet(θs₂)
    catch
        noerrors = false
        lmmlog!(lmm, LMMLogMsg(:ERROR, "logdet(θ₂) not estimated during REML calculation"))
        return (1e100, nothing, nothing, 1e100, noerrors)
    end
    return   θ₁ + logdetθ₂ + θ₃ + c, β, θs₂, θ₃, noerrors #REML, β, iC, θ₃, errors
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
    # Vector log determinant of V matrix
    θ₁            = zero(T)
    θ₂            = zeros(T, lmm.rankx, lmm.rankx)
    θ₃            = zero(T)
    #---------------------------------------------------------------------------
    #q             = zero(Int)
    #qswm          = zero(Int)
    #Vp            = Matrix{T}(undef, 0, 0)
    logdetθ₂      = zero(T)
    akk           = zeros(T, lmm.covstr.maxn + lmm.rankx) #temp for sweep

    @inbounds for i = 1:n
        q    = length(lmm.covstr.vcovblock[i])
        qswm = q + lmm.rankx
        Vp  = zeros(T, q + lmm.rankx, q + lmm.rankx)
        V   = view(Vp, 1:q, 1:q)
        Vx   = view(Vp, 1:q, q+1:q+lmm.rankx)
        Vx  .= data.xv[i]
        vmatrix!(V, θ, lmm, i)
        #-----------------------------------------------------------------------
        try
            θ₁  += logdetv(Symmetric(V))
        catch
            lmmlog!(lmm, LMMLogMsg(:ERROR, "θ₁ not estimated during REML calculation, V isn't positive definite or |V| less zero."))
            return (1e100, nothing, nothing, 1e100)
        end
        #-----------------------------------------------------------------------
        #provide zeros
        sweepb!(fill!(view(akk, 1:qswm), zero(T)), Vp, 1:q)
        #sweep!(Vp, 1:q)
        #V⁻¹ = Symmetric(utriaply!(x -> -x, V))
        V⁻¹ = Symmetric(V)
        #-----------------------------------------------------------------------
        #qswm = size(Vp, 1)
        #θ₂ .-= Symmetric(view(Vp, q + 1:qswm, q + 1:qswm))
        subutri!(θ₂, view(Vp, q + 1:qswm, q + 1:qswm))
        #-----------------------------------------------------------------------
        @inbounds θ₃  += mulθ₃(data.yv[i], data.xv[i], β, V⁻¹)
    end
    θs₂ = Symmetric(θ₂)
    try
        logdetθ₂ = logdet(θs₂)
    catch
        return (1e100, nothing, nothing, 1e100)
    end
    return   θ₁ + logdetθ₂ + θ₃ + c, θs₂, θ₃ #REML, iC, θ₃
end
################################################################################
#                     REML AI-like / scoring part
################################################################################
function sweep_ai(lmm, data::AbstractLMMDataBlocks, θ::Vector{T}, β::Vector) where T <: Number
    n             = length(lmm.covstr.vcovblock)
    N             = length(lmm.data.yv)
    c             = (N - lmm.rankx)*log(2π)
    θ₃            = zero(T)
    #q             = zero(Int)
    #qswm          = zero(Int)
    #Vp            = Matrix{T}(undef, 0, 0)
    akk           = zeros(T, lmm.covstr.maxn + lmm.rankx) #temp for sweep

    @inbounds for i = 1:n
        q    = length(lmm.covstr.vcovblock[i])
        qswm = q + lmm.rankx
        Vp  = zeros(T, q + lmm.rankx, q + lmm.rankx)
        V   = view(Vp, 1:q, 1:q)
        Vx   = view(Vp, 1:q, q+1:q+lmm.rankx)
        Vx  .= data.xv[i]
        vmatrix!(V, θ, lmm, i)
        sweepb!(fill!(view(akk, 1:qswm), zero(T)), Vp, 1:q)
        #sweep!(Vp, 1:q)
        V⁻¹ = Symmetric(V)
        @inbounds θ₃  += mulθ₃(data.yv[i], data.xv[i], β, V⁻¹)
    end
    return  θ₃
end
function sweep_score(lmm, data::AbstractLMMDataBlocks, θ::Vector{T}, β::Vector) where T <: Number
    n             = length(lmm.covstr.vcovblock)
    #---------------------------------------------------------------------------
    # Vector log determinant of V matrix
    θ₁            = zero(T)
    θ₃            = zero(T)
    #---------------------------------------------------------------------------
    q             = zero(Int)
    #qswm          = zero(Int)
    #Vp            = Matrix{T}(undef, 0, 0)
    akk           = zeros(T, lmm.covstr.maxn + lmm.rankx) #temp for sweep

    @inbounds for i = 1:n
        q    = length(lmm.covstr.vcovblock[i])
        qswm = q + lmm.rankx
        Vp  = zeros(T, q + lmm.rankx, q + lmm.rankx)
        V   = view(Vp, 1:q, 1:q)
        Vx   = view(Vp, 1:q, q+1:q+lmm.rankx)
        Vx  .= data.xv[i]
        vmatrix!(V, θ, lmm, i)
        #-----------------------------------------------------------------------
        try
            θ₁  += logdetv(Symmetric(V))
        catch
            return nothing
        end
        sweepb!(fill!(view(akk, 1:qswm), zero(T)), Vp, 1:q)
        #sweep!(Vp, 1:q)
        V⁻¹ = Symmetric(V)
        @inbounds θ₃  += mulθ₃(data.yv[i], data.xv[i], β, V⁻¹)
    end
    return   -θ₁ + θ₃
end
################################################################################
#                     β calculation
################################################################################
function sweep_β(lmm, data::AbstractLMMDataBlocks, θ::Vector{T}) where T <: Number
    n             = length(lmm.covstr.vcovblock)
    N             = length(lmm.data.yv)
    c             = (N - lmm.rankx)*log(2π)
    #---------------------------------------------------------------------------
    θ₂            = zeros(T, lmm.rankx, lmm.rankx)
    βm            = zeros(T, lmm.rankx)
    β             = Vector{T}(undef, lmm.rankx)
    #---------------------------------------------------------------------------
    #q             = zero(Int)
    #qswm          = zero(Int)
    #Vp            = Matrix{T}(undef, 0, 0)
    akk           = zeros(T, lmm.covstr.maxn + lmm.rankx) #temp for sweep

    @inbounds for i = 1:n
        q    = length(lmm.covstr.vcovblock[i])
        qswm = q + lmm.rankx
        Vp  = zeros(T, q + lmm.rankx, q + lmm.rankx)
        V   = view(Vp, 1:q, 1:q)
        Vx   = view(Vp, 1:q, q+1:q+lmm.rankx)
        Vx  .= data.xv[i]
        vmatrix!(V, θ, lmm, i)
        #-----------------------------------------------------------------------
        sweepb!(fill!(view(akk, 1:qswm), zero(T)), Vp, 1:q)
        #sweep!(Vp, 1:q)
        #-----------------------------------------------------------------------
        #qswm = size(Vp, 1)
        #θ₂ .-= Symmetric(view(Vp, q + 1:qswm, q + 1:qswm))
        subutri!(θ₂, view(Vp, q + 1:qswm, q + 1:qswm))
        mulαtβinc!(βm, view(Vp, 1:q, q + 1:qswm), data.yv[i])
        #-----------------------------------------------------------------------
    end
    mul!(β, inv(Symmetric(θ₂)), βm)
    return  β
end
################################################################################
#                     variance-covariance matrix of β
################################################################################
function sweep_β_cov(lmm, θ::Vector{T}, β::Vector) where T <: Number
    n             = length(lmm.covstr.vcovblock)
    N             = length(lmm.data.yv)
    c             = (N - lmm.rankx)*log(2π)
    #---------------------------------------------------------------------------
    θ₂            = zeros(T, lmm.rankx, lmm.rankx)
    #---------------------------------------------------------------------------
    #q             = zero(Int)
    #qswm          = zero(Int)
    #Vp            = Matrix{T}(undef, 0, 0)
    akk           = zeros(T, lmm.covstr.maxn + lmm.rankx) #temp for sweep

    @inbounds for i = 1:n
        q    = length(lmm.covstr.vcovblock[i])
        qswm = q + lmm.rankx
        Vp  = zeros(T, q + lmm.rankx, q + lmm.rankx)
        V   = view(Vp, 1:q, 1:q)
        Vx   = view(Vp, 1:q, q+1:q+lmm.rankx)
        Vx  .= view(lmm.data.xv,  lmm.covstr.vcovblock[i],:)
        vmatrix!(V, θ, lmm, i)
        #-----------------------------------------------------------------------
        sweepb!(fill!(view(akk, 1:qswm), zero(T)), Vp, 1:q)
        #sweep!(Vp, 1:q)
        #V⁻¹ = Symmetric(utriaply!(x -> -x, V))
        #-----------------------------------------------------------------------
        #qswm = size(Vp, 1)
        #θ₂ .-= Symmetric(view(Vp, q + 1:qswm, q + 1:qswm))
        subutri!(θ₂, view(Vp, q + 1:qswm, q + 1:qswm))
        #-----------------------------------------------------------------------
    end
    return Symmetric(θ₂)
end
