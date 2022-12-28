###############################################################################
#                       G MATRIX FUNCTIONS
################################################################################

################################################################################
@noinline function gmatvec(θ::Vector{T}, covstr) where T
    gt = [Symmetric(zeros(T, covstr.q[r], covstr.q[r])) for r in 1:covstr.rn]
    for r = 1:covstr.rn
        if covstr.random[r].covtype.z
            gmat!(gt[r].data, view(θ, covstr.tr[r]), covstr.random[r].covtype.s)
        end
    end
    gt
end
# Main
@noinline function zgz_base_inc!(mx::AbstractArray, G, covstr, bi)
    block = covstr.vcovblock[bi]
    if covstr.random[1].covtype.z
        for r = 1:covstr.rn
            zblock    = view(covstr.z, block, covstr.zrndur[r])
            @inbounds for i = 1:subjn(covstr, r, bi)
                sb = getsubj(covstr, r, bi, i)
                mulαβαtinc!(view(mx, sb, sb), view(zblock, sb, :), G[r])
            end
        end
    end
    mx
end

################################################################################
#=
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
=#
#=
function gmat_switch!(G, θ, covstr, r)
    gmat!(G, view(θ, covstr.tr[r]), covstr.random[r].covtype.s)
    G
end
=#
################################################################################
#SI
function gmat!(::Any, ::Any, ::AbstractCovarianceType)
    error("No gmat! method defined for thit structure!")
end
function gmat!(mx, ::Any, ::ZERO)
    mx
end
#SI
Base.@propagate_inbounds function gmat!(mx, θ, ::SI_)
    val = θ[1] ^ 2
    @inbounds @simd for i = 1:size(mx, 1)
        mx[i, i] = val
    end
    mx
end
#DIAG
function gmat!(mx, θ, ::DIAG_)
    @inbounds @simd for i = 1:size(mx, 1)
        mx[i, i] = θ[i] ^ 2
    end
    mx
end
#AR
function gmat!(mx, θ, ::AR_)
    de  = θ[1] ^ 2
    s   = size(mx, 1)
    @inbounds @simd for i = 1:s
        mx[i, i] = de
    end
    if s > 1
        for m = 1:s - 1
            @inbounds @simd for n = m + 1:s
                mx[m, n] = de * θ[2] ^ (n - m)
            end
        end
    end
    mx
end
#ARH
function gmat!(mx, θ, ::ARH_)
    s = size(mx, 1)
    @inbounds @simd for m = 1:s
        mx[m, m] = θ[m]
    end
    if s > 1
        for m = 1:s - 1
            @inbounds @simd for n = m + 1:s
                mx[m, n] = mx[m, m] * mx[n, n] * θ[end] ^ (n - m)
            end
        end
    end
    @inbounds @simd for m = 1:s
        mx[m, m] = mx[m, m] * mx[m, m]
    end
    mx
end
#CS
function gmat!(mx, θ, ::CS_)
    s = size(mx, 1)
    mx .= θ[1]^2
    if s > 1
        for m = 1:s - 1
            @inbounds @simd for n = m + 1:s
                mx[m, n] = mx[m, m] * θ[2]
            end
        end
    end
    mx
end
#CSH
function gmat!(mx, θ, ::CSH_)
    s = size(mx, 1)
    @inbounds @simd for m = 1:s
        mx[m, m] = θ[m]
    end
    if s > 1
        for m = 1:s - 1
            @inbounds @simd for n = m + 1:s
                mx[m, n] = mx[m, m] * mx[n, n] * θ[end]
            end
        end
    end
    @inbounds @simd for m = 1:s
        mx[m, m] = mx[m, m] * mx[m, m]
    end
    mx
end
################################################################################
#ARMA
function gmat!(mx, θ, ::ARMA_)
    de  = θ[1] ^ 2
    s   = size(mx, 1)
    @inbounds @simd for i = 1:s
        mx[i, i] = de
    end
    if s > 1
        for m = 1:s - 1
            @inbounds @simd for n = m + 1:s
                mx[m, n] = de * θ[2] * θ[3] ^ (n - m - 1)
            end
        end
    end
    mx
end
#TOEP
function gmat!(mx, θ, ::TOEP_)
    de  = θ[1] ^ 2    #diagonal element
    s   = size(mx, 1) #size
    @inbounds @simd for i = 1:s
        mx[i, i] = de
    end
    if s > 1
        for m = 1:s - 1
            @inbounds @simd for n = m + 1:s
                mx[m, n] = de * θ[n-m+1]
            end
        end
    end
    mx
end
function gmat!(mx, θ, ct::TOEPP_)
    de  = θ[1] ^ 2    #diagonal element
    s   = size(mx, 1) #size
    @inbounds @simd for i = 1:s
        mx[i, i] = de
    end
    if s > 1 && ct.p > 1
        for m = 1:s - 1
            @inbounds @simd for n = m + 1:(m + ct.p - 1 > s ? s : m + ct.p - 1)
                mx[m, n] = de * θ[n - m + 1]
            end
        end
    end
    mx
end
#TOEPH
function gmat!(mx, θ, ::TOEPH_)
    s = size(mx, 2)
    @inbounds @simd for m = 1:s
        mx[m, m] = θ[m]
    end
    if s > 1
        for m = 1:s - 1
            @inbounds @simd for n = m + 1:s
                mx[m, n] = mx[m, m] * mx[n, n] * θ[n-m+s]
            end
        end
    end
    @inbounds @simd for m = 1:s
        mx[m, m] = mx[m, m] * mx[m, m]
    end
    mx
end
#TOEPHP
function gmat!(mx, θ, ct::TOEPHP_)
    s = size(mx, 2)
    @inbounds @simd for m = 1:s
        mx[m, m] = θ[m]
    end
    if s > 1 && ct.p > 1
        for m = 1:s - 1
            for n = m + 1:(m + ct.p - 1 > s ? s : m + ct.p - 1)
                @inbounds mx[m, n] = mx[m, m] * mx[n, n] * θ[n - m + s]
            end
        end
    end
    @inbounds @simd for m = 1:s
        mx[m, m] = mx[m, m] * mx[m, m]
    end
    mx
end
#UN
function gmat!(mx, θ, ::UN_)
    s = size(mx, 1)
    @inbounds @simd for m = 1:s
        mx[m, m] = θ[m]
    end
    if s > 1
        for m = 1:s - 1
            @inbounds @simd for n = m + 1:s
                mx[m, n] = mx[m, m] * mx[n, n] * θ[s+tpnum(m, n, s)]
            end
        end
    end
    @inbounds @simd for m = 1:s
        v = mx[m, m]
        mx[m, m] = v * v
    end
    mx
end

function tpnum(m, n, s)
    b = 0
    for i in 1:m
        b += s - i
    end
    b -= s - n
end
#=
function tpnum(m, n, s)
    div(m*(2s - 1 - m), 2) - s + n 
end
=#

################################################################################
# Grads

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

"""
Number of parameters correspondig to random effects (G) in theta vector.
"""
function thetagnum(covstr)
    covstr.tl - covstr.t[end]
end
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

function grad_zgz_base_inc!(mx::AbstractArray, G, gn, covstr, bi)
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

