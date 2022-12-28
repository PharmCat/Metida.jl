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

function checkmatrix!(mx::AbstractMatrix{T}) where T
    e = true
    dm = zero(T)
    @inbounds for  i = 1:size(mx, 1)
        if mx[i,i] > dm dm = mx[i,i] end
    end
    dm *= sqrt(eps())
    @inbounds for i = 1:size(mx, 1)
        if mx[i,i] <= dm
            mx[i,i] = dm
            e = false
        end
    end
    e
end
################################################################################
#                     REML without provided β
################################################################################
function reml_sweep_β(lmm, data, θ::Vector{T}; maxthreads::Int = 16) where T # Main optimization way - make gradient / hessian analytical / semi-analytical functions
    n             = length(lmm.covstr.vcovblock)
    N             = length(lmm.data.yv)
    c             = (N - lmm.rankx)*log(2π)
    p             = size(lmm.data.xv, 2)
    #---------------------------------------------------------------------------
    V⁻¹           = Vector{SubArray{T, 2, Matrix{T}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false}}(undef, n)
    θ₃            = zero(T)
    β             = Vector{T}(undef, p)
    #---------------------------------------------------------------------------
    gvec          = gmatvec(θ, lmm.covstr)
    rθ            = θ[lmm.covstr.tr[end]] # R part of θ
    noerror       = true
        ncore     = min(num_cores(), n, maxthreads)
        accθ₁     = zeros(T, ncore)
        accθ₂     = Vector{Matrix{T}}(undef, ncore)
        accβm     = Vector{Vector{T}}(undef, ncore)
        swtw      = Vector{Vector{T}}(undef, ncore)
        erroracc  = trues(ncore)
        d, r      = divrem(n, ncore)
        Base.Threads.@threads for t = 1:ncore
            # Vp - matrix for sweep operation
            # [V  X
            #  X' 0]
            offset   = min(t - 1, r) + (t - 1)*d
            accθ₂[t] = zeros(T, p, p)
            accβm[t] = zeros(T, p)
            swtw[t]  = zeros(T, lmm.maxvcbl)
            @inbounds for j ∈ 1:d + (t ≤ r)
                i    =  offset + j
                q    = length(lmm.covstr.vcovblock[i])
                qswm = q + lmm.rankx
                Vp   = zeros(T, qswm, qswm)
                V    = view(Vp, 1:q, 1:q)
                Vx   = view(Vp, 1:q, q+1:qswm)
                Vc   = view(Vp, q+1:qswm, q+1:qswm)
                copyto!(Vx, data.xv[i])
            #-------------------------------------------------------------------
            # Make V matrix
                vmatrix!(V, gvec, rθ, lmm, i)
            #-----------------------------------------------------------------------
                if length(swtw[t]) != qswm resize!(swtw[t], qswm) end
                swm, swr, ne  = sweepb!(swtw[t], Vp, 1:q; logdet = true)
                V⁻¹[i] = V
            #-----------------------------------------------------------------------
                if ne == false erroracc[t] = false end
                accθ₁[t] += swr
                accθ₂[t] .-= Vc
                mulαtβinc!(accβm[t], Vx, data.yv[i])
            end
            #-----------------------------------------------------------------------
        end
        θ₁      = sum(accθ₁)
        θ₂      = sum(accθ₂)
        βm      = sum(accβm)
        noerror = all(erroracc)
        noerror = noerror * checkmatrix!(θ₂)
        θs₂     = Symmetric(θ₂)
        # Cholesky decomposition for matrix inverse θs₂ - Symmetric(θ₂); C = θ₂⁻¹
        cθs₂    = cholesky(θs₂, check = false)
        if issuccess(cθs₂)
            # β calculation
            mul!(β, inv(cθs₂), βm)
            # final θ₂
            logdetθ₂ = logdet(cθs₂)
            # θ₃
            @inbounds @simd for i = 1:n
                θ₃ += mulθ₃(data.yv[i], data.xv[i], β, V⁻¹[i])
            end
        else
            β       .= NaN
            return   Inf, β, Inf, Inf, false
        end
    return   θ₁ + logdetθ₂ + θ₃ + c, β, θs₂, θ₃, noerror #REML, β, iC, θ₃, errors
end
# Using BLAS, LAPACK - non ForwardDiff, used by MetidaNLopt
function reml_sweep_β_nlopt(lmm, data, θ::Vector{T}; maxthreads::Int = 16) where T
    n             = length(lmm.covstr.vcovblock)
    N             = length(lmm.data.yv)
    c             = (N - lmm.rankx)*log(2π)
    #---------------------------------------------------------------------------
    θ₁            = zero(T)
    θ₂            = zeros(T, lmm.rankx, lmm.rankx)
    θ₃            = zero(T)
    A             = Vector{Matrix{T}}(undef, n)
    logdetθ₂      = zero(T)
    gvec          = gmatvec(θ, lmm.covstr)
    rθ            = θ[lmm.covstr.tr[end]] # R part of θ
    noerror       = true
        ncore     = min(num_cores(), n, maxthreads)
        accθ₁     = zeros(T, ncore)
        accθ₂     = Vector{Matrix{T}}(undef, ncore)
        accβm     = Vector{Vector{T}}(undef, ncore)
        erroracc  = trues(ncore)
        d, r = divrem(n, ncore)
        Base.Threads.@threads for t = 1:ncore
            offset   = min(t-1, r) + (t-1)*d
            accθ₂[t] = zeros(T, lmm.rankx, lmm.rankx)
            accβm[t] = zeros(T, lmm.rankx)
            @inbounds for j ∈ 1:d+(t ≤ r)
                i =  offset + j
                q    = length(lmm.covstr.vcovblock[i])
                V    = zeros(T, q, q)
                vmatrix!(V, gvec, rθ, lmm, i)
        #-------------------------------------------------------------------
        # Cholesky
                Ai, info = LinearAlgebra.LAPACK.potrf!('U', V)
                A[i] = Ai
                vX   = LinearAlgebra.LAPACK.potrs!('U', Ai, copy(data.xv[i]))
                vy   = LinearAlgebra.LAPACK.potrs!('U', Ai, copy(data.yv[i]))
            # Check matrix and make it avialible for logdet computation
                if info == 0
                    θ₁ld = logdet(Cholesky(Ai, 'U', 0))
                else
                    erroracc[t] = false
                    break
                end
                accθ₁[t]  += θ₁ld
                mul!(accθ₂[t], data.xv[i]', vX, 1, 1)
                mul!(accβm[t], data.xv[i]', vy, 1, 1)
            end
        #-------------------------------------------------------------------
        end
        noerror = all(erroracc)
        if !noerror
            β = fill(NaN, lmm.rankx)
            return   Inf, β, Inf, Inf, false
        end
        θ₁   = sum(accθ₁)
        θ₂tc = sum(accθ₂)
        βtc  = sum(accβm)
    # Beta calculation
        copyto!(θ₂, θ₂tc)
        ldθ₂, info = LinearAlgebra.LAPACK.potrf!('U', θ₂tc)
        if info != 0
            β = fill(NaN, lmm.rankx)
            return   Inf, β, Inf, Inf, false
        end
        LinearAlgebra.LAPACK.potrs!('U', θ₂tc, βtc)
        β = βtc
    # θ₂ calculation
        logdetθ₂ = logdet(Cholesky(ldθ₂, 'U', 0))
    # θ₃ calculation
        @inbounds @simd for i = 1:n
            r    = mul!(copy(data.yv[i]), data.xv[i], βtc, -1, 1)
            vr   = LinearAlgebra.LAPACK.potrs!('U', A[i], copy(r))
            θ₃  += dot(r, vr)
        end
    return   θ₁ + logdetθ₂ + θ₃ + c, β, θ₂, θ₃, noerror
end
################################################################################
#                     REML with provided β
################################################################################
function core_sweep_β(lmm, data, θ::Vector{T}, β, n; maxthreads::Int = 16) where T
    ncore     = min(num_cores(), n, maxthreads)
    accθ₁     = zeros(T, ncore)
    accθ₂     = Vector{Matrix{T}}(undef, ncore)
    accθ₃     = zeros(T, ncore)
    erroracc  = trues(ncore)
    gvec      = gmatvec(θ, lmm.covstr)
    rθ        = θ[lmm.covstr.tr[end]] # R part of θ
    d, r = divrem(n, ncore)
    Base.Threads.@threads for t = 1:ncore
        offset = min(t-1, r) + (t-1)*d
        accθ₂[t] = zeros(T, lmm.rankx, lmm.rankx)
        @inbounds for j ∈ 1:d+(t ≤ r)
            i =  offset + j
            q    = length(lmm.covstr.vcovblock[i])
            qswm = q + lmm.rankx
            Vp   = zeros(T, qswm, qswm)
            V    = view(Vp, 1:q, 1:q)
            Vx   = view(Vp, 1:q, q+1:qswm)
            Vc   = view(Vp, q+1:qswm, q+1:qswm)
            copyto!(Vx, data.xv[i])
            vmatrix!(V, gvec, rθ, lmm, i)
            #-----------------------------------------------------------------------
            swm, swr, ne = sweepb!(Vector{T}(undef, qswm), Vp, 1:q; logdet = true)
            #-----------------------------------------------------------------------
            if ne == false erroracc[t] = false end
            accθ₁[t] += swr
            accθ₂[t] .-= Vc
            accθ₃[t]  += mulθ₃(data.yv[i], data.xv[i], β, V)
        end
    end
    sum(accθ₁), sum(accθ₂), sum(accθ₃), all(erroracc)
end
###
function reml_sweep_β(lmm, data, θ::Vector{T}, β; kwargs...) where T
    n             = length(lmm.covstr.vcovblock)
    N             = length(lmm.data.yv)
    c             = (N - lmm.rankx)*log(2π)
    θ₁, θ₂, θ₃, noerror = core_sweep_β(lmm, data, θ::Vector{T}, β, n; kwargs...)
    θs₂      = Symmetric(θ₂)
    logdetθ₂ = logdet(θs₂)
    return   θ₁ + logdetθ₂ + θ₃ + c, θs₂, θ₃, noerror #REML, iC, θ₃
end
################################################################################
#                     REML AI-like / scoring part
################################################################################
function sweep_ai(lmm, data, θ, β; kwargs...)
    n                   = length(lmm.covstr.vcovblock)
    θ₁, θ₂, θ₃, noerror = core_sweep_β(lmm, data, θ, β, n; kwargs...)
    return  θ₃
end
function sweep_score(lmm, data, θ, β; kwargs...)
    n                   = length(lmm.covstr.vcovblock)
    θ₁, θ₂, θ₃, noerror = core_sweep_β(lmm, data, θ, β, n; kwargs...)
    return -θ₁ + θ₃
end
################################################################################
#                     variance-covariance matrix of β
################################################################################
function sweep_β_cov(lmm, data, θ, β; kwargs...)
    n                   = length(lmm.covstr.vcovblock)
    θ₁, θ₂, θ₃, noerror = core_sweep_β(lmm, data, θ, β, n; kwargs...)
    return Symmetric(θ₂)
end
