#reml.jl
function subutri!(a, b)
    s = size(a,1)
    if s == 1 return a[1,1] -= b[1,1] end
    @simd for m = 1:s
        @simd for n = m:s
            @inbounds a[m,n] -= b[m,n]
        end
    end
    a
end

function fillzeroutri!(a::AbstractArray{T})  where T
    tr = UpperTriangular(a)
    fill!(tr, zero(T))
end

################################################################################
#                     REML without provided β
################################################################################
function reml_sweep_β(lmm, θ::Vector{T}; syrkblas::Bool = false) where T <: Number
    data = LMMDataViews(lmm)
    reml_sweep_β(lmm, data, θ; syrkblas = syrkblas)
end
function reml_sweep_β(lmm, data::AbstractLMMDataBlocks, θ::Vector{T}; syrkblas::Bool = false) where T <: Number
    n             = length(lmm.covstr.vcovblock)
    N             = length(lmm.data.yv)
    c             = (N - lmm.rankx)*log(2π)
    #---------------------------------------------------------------------------
    V⁻¹           = Vector{AbstractArray{T}}(undef, n)
    θ₃            = zero(T)
    β             = Vector{T}(undef, lmm.rankx)
    #---------------------------------------------------------------------------
    #logdetθ₂      = zero(T)
    noerror       = true
        ncore     = min(num_cores(), n)
        accθ₁     = zeros(T, ncore)
        accθ₂     = Vector{Matrix{T}}(undef, ncore)
        accβm     = Vector{Vector{T}}(undef, ncore)
        erroracc  = trues(ncore)
        d, r = divrem(n, ncore)
        Base.Threads.@threads for t = 1:ncore
        #@batch for t = 1:ncore
        #for t = 1:ncore
            # Vp - matrix for sweep operation
            # [V  X
            #  X' 0]
            offset = min(t-1, r) + (t-1)*d
            accθ₂[t] = zeros(T, lmm.rankx, lmm.rankx)
            accβm[t] = zeros(T, lmm.rankx)

            @inbounds for j ∈ 1:d+(t ≤ r)
                i =  offset + j
                q    = length(lmm.covstr.vcovblock[i])
                qswm = q + lmm.rankx
                Vp   = Matrix{T}(undef, qswm, qswm)
                V    = view(Vp, 1:q, 1:q)
                Vx   = view(Vp, 1:q, q+1:qswm)
                Vc   = view(Vp, q+1:qswm, q+1:qswm)
                fillzeroutri!(V)
                copyto!(Vx, data.xv[i])
                fillzeroutri!(Vc)
            #-------------------------------------------------------------------
            # Make V matrix
                vmatrix!(V, θ, lmm, i)
            #-----------------------------------------------------------------------
                swm, swr, ne  = sweepb!(Vector{T}(undef, qswm), Vp, 1:q; logdet = true, syrkblas = syrkblas)
                V⁻¹[i] = V
            #-----------------------------------------------------------------------
                if ne == false erroracc[t] = false end
                accθ₁[t] += swr
                subutri!(accθ₂[t], Vc)
                mulαtβinc!(accβm[t], Vx, data.yv[i])
            end
            #-----------------------------------------------------------------------
        end
        θ₁      = sum(accθ₁)
        θ₂      = sum(accθ₂)
        βm      = sum(accβm)
        noerror = all(erroracc)
        θs₂     = Symmetric(θ₂)
        # Cholesky decomposition for matrix inverse θs₂ - Symmetric(θ₂); C = θ₂⁻¹
        cθs₂    = cholesky(θs₂)
        # β calculation
        mul!(β, inv(cθs₂), βm)
        # θ₃
        @inbounds @simd for i = 1:n
            θ₃ += mulθ₃(data.yv[i], data.xv[i], β, V⁻¹[i])
        end
        # final θ₂
        logdetθ₂ = logdet(cθs₂)
    return   θ₁ + logdetθ₂ + θ₃ + c, β, θs₂, θ₃, noerror #REML, β, iC, θ₃, errors
end
################################################################################
#                     REML with provided β
################################################################################
function reml_sweep_β(lmm, θ::Vector{T}, β::Vector) where T <: Number
    data = LMMDataViews(lmm)
    reml_sweep_β(lmm, data, θ, β)
end

function core_sweep_β(lmm, data, θ::Vector{T}, β, n) where T
    ncore     = min(num_cores(), n)
    accθ₁     = zeros(T, ncore)
    accθ₂     = Vector{Matrix{T}}(undef, ncore)
    accθ₃     = zeros(T, ncore)
    erroracc  = trues(ncore)
    d, r = divrem(n, ncore)
    Base.Threads.@threads for t = 1:ncore
    #@batch for t = 1:ncore
    #for t = 1:ncore
        offset = min(t-1, r) + (t-1)*d
        accθ₂[t] = zeros(T, lmm.rankx, lmm.rankx)
        @inbounds for j ∈ 1:d+(t ≤ r)
            i =  offset + j
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
            swm, swr, ne = sweepb!(Vector{T}(undef, qswm), Vp, 1:q; logdet = true)
            #-----------------------------------------------------------------------
            if ne == false erroracc[t] = false end
            accθ₁[t] += swr
            subutri!(accθ₂[t], Vc)
            accθ₃[t]  += mulθ₃(data.yv[i], data.xv[i], β, V)
        end
    end
    sum(accθ₁), sum(accθ₂), sum(accθ₃), all(erroracc)
end

function reml_sweep_β(lmm, data::AbstractLMMDataBlocks, θ::Vector{T}, β::Vector) where T <: Number
    n             = length(lmm.covstr.vcovblock)
    N             = length(lmm.data.yv)
    c             = (N - lmm.rankx)*log(2π)
    θ₁, θ₂, θ₃, noerror = core_sweep_β(lmm, data, θ::Vector{T}, β, n)
    θs₂      = Symmetric(θ₂)
    logdetθ₂ = logdet(θs₂)
    return   θ₁ + logdetθ₂ + θ₃ + c, θs₂, θ₃, noerror #REML, iC, θ₃
end
################################################################################
#                     REML AI-like / scoring part
################################################################################
function sweep_ai(lmm, data::AbstractLMMDataBlocks, θ::Vector{T}, β::Vector) where T <: Number
    n                   = length(lmm.covstr.vcovblock)
    θ₁, θ₂, θ₃, noerror = core_sweep_β(lmm, data, θ::Vector{T}, β, n)
    return  θ₃
end
function sweep_score(lmm, data::AbstractLMMDataBlocks, θ::Vector{T}, β::Vector) where T <: Number
    n                   = length(lmm.covstr.vcovblock)
    θ₁, θ₂, θ₃, noerror = core_sweep_β(lmm, data, θ::Vector{T}, β, n)
    return -θ₁ + θ₃
end
################################################################################
#                     variance-covariance matrix of β
################################################################################
function sweep_β_cov(lmm, data::AbstractLMMDataBlocks, θ::Vector{T}, β::Vector) where T <: Number
    n                   = length(lmm.covstr.vcovblock)
    θ₁, θ₂, θ₃, noerror = core_sweep_β(lmm, data, θ::Vector{T}, β, n)
    return Symmetric(θ₂)
end
