# Not included in main module, but can be used later

################################################################################
# Grads
"""
    trmulαβ(A::AbstractMatrix{T}, B::AbstractMatrix) where T

Trace A*B, A and B same size, B - Symmetric, 
"""
function trmulαβ(A::AbstractMatrix{T}, B::AbstractMatrix) where T
    c = zero(T)
    if LinearAlgebra.checksquare(B) != LinearAlgebra.checksquare(A) error("Not equal size!") end
    for m = 1:size(B, 1)
        for n = 1:size(A, 1)
            c += A[n, m] * B[n, m]
        end
    end
    c
end
function lpinv!(A)
    LinearAlgebra.checksquare(A)
    LinearAlgebra.LAPACK.potrf!('U', A)
    LinearAlgebra.LAPACK.potri!('U', A)
end
# NEED β calk and @threads 
function reml_grad(lmm, data, θ::Vector{T}, β; maxthreads::Int = 16)  where T
    n     = length(lmm.covstr.vcovblock)
    tl    = lmm.covstr.tl
    p     = size(lmm.data.xv, 2)
    gvec  = gmatvec(θ, lmm.covstr)
    gradg = grad_gmatvec(θ, lmm.covstr)
    iV    = Vector{Symmetric{T, Matrix{T}}}(undef, n)
    rθ    = θ[lmm.covstr.tr[end]] # R part of θ
    info  = 0
    ncore     = min(num_cores(), n, maxthreads)
    accH      = Vector{Matrix{T}}(undef, ncore)
    d, r      = divrem(n, ncore)
    Base.Threads.@threads for t = 1:ncore
        offset   = min(t - 1, r) + (t - 1)*d
        for j ∈ 1:d+(t ≤ r)
            i       =  offset + j
            q       = length(lmm.covstr.vcovblock[i])
            V       = zeros(q, q)
            vmatrix!(V, gvec, rθ, lmm, i)
            iVi     = Symmetric(lpinv!(V))
            iV[i]   = iVi
            accH[t] = zeros(T, p, p)
            mulαβαtinc!(accH[t], data.xv[i]', iVi)
            #H  += data.xv[i]' * iV[i] * data.xv[i]
        end
    end
    iH        = Symmetric(lpinv!(sum(accH)))
    d, r      = divrem(n, ncore)
    accθ₁     = Vector{Vector{T}}(undef, ncore)
    accθ₂     = Vector{Vector{T}}(undef, ncore) 
    accθ₃     = Vector{Vector{T}}(undef, ncore) 
    Base.Threads.@threads for t = 1:ncore
        accθ₁[t]     = zeros(T, tl)
        accθ₂[t]     = zeros(T, tl)
        accθ₃[t]     = zeros(T, tl)
        offset   = min(t - 1, r) + (t - 1)*d
        for j ∈ 1:d + (t ≤ r)
            i    =  offset + j
            q    = length(lmm.covstr.vcovblock[i])
            rv   = copy(data.yv[i]) 
            mul!(rv, data.xv[i], β, -1, 1)
            jV   = grad_vmatrix(gradg, rθ, lmm, i)
            Aj   = Symmetric(zeros(q, q))
            iVi = iV[i]
            XtAjX   = zeros(p, p) 
            for k = 1:tl
                jVk = jV[k]
                fill!(Aj, zero(T))
                mulαβαtinc!(Aj.data, iVi, jVk)
                #Aj      = iV[i] * jV[j] * iV[i]
                accθ₁[t][k]  += trmulαβ(iVi, jVk)
                #θ₁[j]  += tr(iV[i] * jV[j])
                fill!(XtAjX, zero(T))
                mulαβαtinc!(XtAjX, data.xv[i]', Aj)
                accθ₂[t][k] -= trmulαβ(iH, Symmetric(XtAjX))
                #θ₂[j]  -= tr(iH * data.xv[i]' * Aj * data.xv[i])
                accθ₃[t][k]  -= dot(rv, Aj, rv)
                #θ₃[j]  -= r' * Aj * r
            end
        end
    end
    θ₁ = sum(accθ₁)
    θ₂ = sum(accθ₂)
    θ₃ = sum(accθ₃)
    return @. θ₁ + θ₂ + θ₃
end
"""
Number of parameters correspondig to random effects (G) in theta vector.
"""
function thetagnum(covstr)
    covstr.tl - covstr.t[end]
end
# V matrix
function grad_vmatrix(G, rθ::AbstractVector{T}, lmm, i::Int) where T
    gv = Vector{Symmetric{T, Matrix{T}}}(undef, lmm.covstr.tl)
    for gi = 1:lmm.covstr.tl

        q = length(lmm.covstr.vcovblock[i])
        
        gv[gi] = Symmetric(zeros(T, q, q))
        
        grad_zgz_base_inc!(gv[gi].data, G, gi, lmm.covstr, i)

        grad_rmat_base_inc!(gv[gi].data, rθ, gi, lmm.covstr, i)
    end
    gv
end
# Main grad function
function grad_gmatvec(θ::Vector{T}, covstr) where T
    gi = thetagnum(covstr) # total number of grads for G side
    gt = Vector{Symmetric{T, Matrix{T}}}(undef, gi)
    gn = 1
    for r = 1:covstr.rn
        if covstr.random[r].covtype.z
            for g = 1:covstr.t[r]
                gt[gn] = Symmetric(zeros(T, covstr.q[r], covstr.q[r]))
                gmat_g!(gt[gn].data, view(θ, covstr.tr[r]), g, covstr.random[r].covtype.s)
                gn += 1
            end
        end
    end
    gt
end
#
@noinline function grad_zgz_base_inc!(mx::AbstractArray, G, gn, covstr, bi)
    if gn > thetagnum(covstr) return mx end
    block = covstr.vcovblock[bi]
    if covstr.random[1].covtype.z
        r = covstr.emap[gn]
        zblock    = view(covstr.z, block, covstr.zrndur[r])
        for i = 1:subjn(covstr, r, bi)
            sb = getsubj(covstr, r, bi, i)
            mulαβαtinc!(view(mx, sb, sb), view(zblock, sb, :), G[gn])
        end
    end
    mx
end
#
@noinline function grad_rmat_base_inc!(mx, θ, gn, covstr, bi)
    tgn =  thetagnum(covstr)
    if gn <= tgn return mx end
    rn        = gn - tgn
    en        = covstr.rn + 1
    block     = covstr.vcovblock[bi]
    zblock    = view(covstr.rz, block, :)
    @simd for i = 1:subjn(covstr, en, bi)
        sb = getsubj(covstr, en, bi, i)
        rmat_g!(view(mx, sb, sb), θ, view(zblock, sb, :), rn, covstr.repeated.covtype.s)
    end
    mx
end

### G side
###############################################################################
###############################################################################
###############################################################################
function gmat_g!(mx, θ, g::Int, ct::AbstractCovarianceType)
    T = ForwardDiff.Dual{Nothing, Float64, 1}
    gθ  = Vector{T}(undef, length(θ))
    for i = 1:length(θ)
        if i == g gθ[i] = ForwardDiff.Dual(θ[i], 1) else gθ[i] = ForwardDiff.Dual(θ[i], 0) end
    end
    gmx = zeros(T, size(mx))
    Metida.gmat!(gmx, gθ, ct)
    for n = 1:size(gmx, 2)
        for m = 1:n
            mx[m, n] = ForwardDiff.partials(gmx[m, n])[1]
        end
    end
    mx
end
#ZERO
function gmat_g!(mx, ::Any, ::Int, ::ZERO)
    mx
end
#SI
function gmat_g!(mx, θ, ::Int, ::SI_)
    val = 2θ[1]
    @inbounds @simd for i = 1:size(mx, 1)
        mx[i, i] = val
    end
    mx
end
#DIAG
function gmat_g!(mx, θ, g::Int, ::DIAG_)
    for i = 1:size(mx, 1)
        if i == g mx[i, i] = 2θ[i] end
    end
    mx
end

### R side
###############################################################################
###############################################################################
###############################################################################
function rmat_g!(mx, θ, rz::AbstractMatrix, g::Int, ct::AbstractCovarianceType)
    T = ForwardDiff.Dual{Nothing, Float64, 1}
    gθ  = Vector{T}(undef, length(θ))
    for i = 1:length(θ)
        if i == g gθ[i] = ForwardDiff.Dual(θ[i], 1) else gθ[i] = ForwardDiff.Dual(θ[i], 0) end
    end
    gmx = zeros(T, size(mx))
    Metida.rmat!(gmx, gθ, rz, ct)
    for n = 1:size(gmx, 2)
        for m = 1:n
            mx[m, n] = ForwardDiff.partials(gmx[m, n])[1]
        end
    end
    mx
end
#SI
function rmat_g!(mx, θ, ::AbstractMatrix, ::Int, ::SI_)
    val = 2θ[1]
    for i ∈ axes(mx, 1)
            mx[i, i] += val
    end
    mx
end
#DIAG
function rmat_g!(mx, θ, rz, g::Int, ::DIAG_)
    for i ∈ axes(mx, 1)
        for c ∈ axes(θ, 1)
            if c == g mx[i, i] += rz[i, c] * 2θ[c] end
        end
    end
    mx
end
