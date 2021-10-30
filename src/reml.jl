#reml.jl
 Base.@propagate_inbounds function subutri!(a, b)
    s = size(a,1)
    if s == 1 return a[1,1] -= b[1,1] end
    @simd for m = 1:s
        @simd for n = m:s
            a[m,n] -= b[m,n]
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
    θ₁            = zero(T)
    θ₂            = zeros(T, lmm.rankx, lmm.rankx)
    θ₃            = zero(T)
    βm            = zeros(T, lmm.rankx)
    β             = Vector{T}(undef, lmm.rankx)
    #---------------------------------------------------------------------------
    logdetθ₂      = zero(T)
    θs₂           = Symmetric(θ₂)
    #θ₂ut          = UpperTriangular(θ₂)
    #akk = Vector{T}(undef, lmm.covstr.maxn + lmm.rankx) #temp for sweep
    noerror       = true

        ncore     = min(num_cores(), n)
        accθ₁     = zeros(T, ncore)
        accθ₂     = Vector{Matrix{T}}(undef, ncore)
        accβm     = Vector{Vector{T}}(undef, ncore)
        erroracc  = trues(ncore)
        d, r = divrem(n, ncore)
        Base.Threads.@threads for t = 1:ncore

        #@batch per=core for i = 1:n #@fastmath
            # Vp - matrix for sweep operation
            # [V  X
            #  X' 0]
            offset = min(t-1, r) + (t-1)*d
            accθ₂[t] = zeros(T, lmm.rankx, lmm.rankx)
            accβm[t] = zeros(T, lmm.rankx)

            for j ∈ 1:d+(t ≤ r)
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
            #swr  = sweepb!(view(akk, 1:qswm), Vp, 1:q; logdet = true, syrkblas = syrkblas)
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
        θ₁ = sum(accθ₁)
        map(x->θ₂ .+= x, accθ₂) # make upper triangular use only!
        map(x->βm .+= x, accβm)
        noerror = all(erroracc)
        # Cholesky decomposition for matrix inverse θs₂ - Summetric(θ₂); C = θ₂⁻¹
        cθs₂ = cholesky(θs₂)
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
    #Vm            = Matrix{T}(undef, lmm.covstr.maxn + lmm.rankx, lmm.covstr.maxn + lmm.rankx) #!!
    θs₂           = Symmetric(θ₂)
    noerror       = true

        l = Base.Threads.SpinLock()
        Base.Threads.@threads for i = 1:n
            q    = length(lmm.covstr.vcovblock[i])
            qswm = q + lmm.rankx
            Vp   = Matrix{T}(undef, qswm, qswm)
            V    = view(Vp, 1:q, 1:q)
            Vx   = view(Vp, 1:q, q+1:qswm)
            Vc   = view(Vp, q+1:qswm, q+1:qswm)
            #Vp   = view(Vm, 1:q + lmm.rankx, 1:q + lmm.rankx)
            #V    = view(Vm, 1:q, 1:q)
            #Vx   = view(Vm, 1:q, q+1:q+lmm.rankx)
            #Vc   = view(Vm, q + 1:qswm, q + 1:qswm)
            fillzeroutri!(V)
            copyto!(Vx, data.xv[i])
            fillzeroutri!(Vc)
            vmatrix!(V, θ, lmm, i)
            #-----------------------------------------------------------------------
            swm, swr, ne = sweepb!(Vector{T}(undef, qswm), Vp, 1:q; logdet = true)
            #-----------------------------------------------------------------------
            θ₃t = mulθ₃(data.yv[i], data.xv[i], β, V)
            lock(l) do
                if ne == false noerror = false end
                θ₁  += swr
                subutri!(θ₂, view(Vp, q + 1:qswm, q + 1:qswm))
                θ₃  += θ₃t
            end
        end
        logdetθ₂ = logdet(θs₂)

    return   θ₁ + logdetθ₂ + θ₃ + c, θs₂, θ₃, noerror #REML, iC, θ₃
end
################################################################################
#                     REML AI-like / scoring part
################################################################################
function sweep_ai(lmm, data::AbstractLMMDataBlocks, θ::Vector{T}, β::Vector) where T <: Number
    n             = length(lmm.covstr.vcovblock)
    θ₃            = zero(T)
    #akk           = zeros(T, lmm.covstr.maxn + lmm.rankx) #temp for sweep
    #Vm            = Matrix{T}(undef, lmm.covstr.maxn + lmm.rankx, lmm.covstr.maxn + lmm.rankx) #!!
    l = Base.Threads.SpinLock()
    @inbounds Base.Threads.@threads for i = 1:n
        q    = length(lmm.covstr.vcovblock[i])
        qswm = q + lmm.rankx
        Vp   = Matrix{T}(undef, qswm, qswm)
        V    = view(Vp, 1:q, 1:q)
        Vx   = view(Vp, 1:q, q+1:qswm)
        Vc   = view(Vp, q+1:qswm, q+1:qswm)
        #Vp   = view(Vm, 1:q + lmm.rankx, 1:q + lmm.rankx)
        #V    = view(Vm, 1:q, 1:q)
        #Vx   = view(Vm, 1:q, q+1:q+lmm.rankx)
        #Vc   = view(Vm, q + 1:qswm, q + 1:qswm)
        fillzeroutri!(V)
        copyto!(Vx, data.xv[i])
        fillzeroutri!(Vc)
        vmatrix!(V, θ, lmm, i)
        sweepb!(Vector{T}(undef, qswm), Vp, 1:q)
        θ₃t = mulθ₃(data.yv[i], data.xv[i], β, V)
        lock(l) do
            θ₃  += θ₃t
        end
    end
    return  θ₃
end
function sweep_score(lmm, data::AbstractLMMDataBlocks, θ::Vector{T}, β::Vector) where T <: Number
    n             = length(lmm.covstr.vcovblock)
    θ₁            = zero(T)
    θ₃            = zero(T)
    #---------------------------------------------------------------------------
    #akk           = zeros(T, lmm.covstr.maxn + lmm.rankx) #temp for sweep
    #Vm            = Matrix{T}(undef, lmm.covstr.maxn + lmm.rankx, lmm.covstr.maxn + lmm.rankx) #!!
    l = Base.Threads.SpinLock()
    @inbounds Base.Threads.@threads for i = 1:n
        q    = length(lmm.covstr.vcovblock[i])
        qswm = q + lmm.rankx
        Vp   = Matrix{T}(undef, qswm, qswm)
        V    = view(Vp, 1:q, 1:q)
        Vx   = view(Vp, 1:q, q+1:qswm)
        Vc   = view(Vp, q+1:qswm, q+1:qswm)

        #Vp   = view(Vm, 1:q + lmm.rankx, 1:q + lmm.rankx)
        #V    = view(Vm, 1:q, 1:q)
        #Vx   = view(Vm, 1:q, q+1:q+lmm.rankx)
        #Vc   = view(Vm, q + 1:qswm, q + 1:qswm)

        fillzeroutri!(V)
        copyto!(Vx, data.xv[i])
        fillzeroutri!(Vc)
        vmatrix!(V, θ, lmm, i)
        #-----------------------------------------------------------------------
        swr  = sweepb!(Vector{T}(undef, qswm), Vp, 1:q; logdet = true)

        θ₃t  = mulθ₃(data.yv[i], data.xv[i], β, V)
        lock(l) do
            θ₁  += swr[2]
            θ₃  += θ₃t
        end
    end

    return   -θ₁ + θ₃
end

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
