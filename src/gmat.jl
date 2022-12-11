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
function gmat!(::Any, ::Any, ::ZERO)
    nothing
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

################################################################################

# Grads
#=
function sim_gmat(s, θ::AbstractVector{T}, ct::AbstractCovarianceType) where T
    mxt = zeros(T, s, s)
    Symmetric(gmat!(mxt, θ, ct))
end

function grad_gmat!(s, θ, ct::AbstractCovarianceType)
    mxg = zeros(s, s, length(θ))
    gm  = ForwardDiff.jacobian!(mxg, x-> sim_gmat(s, x, ct), θ)
    mxg
end

function grad_gmatvec(θ::Vector{T}, covstr) where T
    gt = [Array{T}(undef, covstr.q[r], covstr.q[r], length(θ)) for r in 1:covstr.rn] #<: fix to vec
    for r = 1:covstr.rn
        if covstr.random[r].covtype.z
            gt[r] = grad_gmat!(covstr.q[r], view(θ, covstr.tr[r]), covstr.random[r].covtype.s)
        end
    end
    gt
end

function grad_zgz_base_inc!(mx::AbstractArray, G, θ::AbstractArray{T}, covstr, block, sblock) where T
    if covstr.random[1].covtype.z
        gl = size(mx, 3)
        for r = 1:covstr.rn
            zblock    = view(covstr.z, block, covstr.zrndur[r])
            @inbounds for i = 1:length(sblock[r])
                for gn = 1:gl
                    mulαβαtinc!(view(mx, sblock[r][i], sblock[r][i], gn), view(zblock, sblock[r][i], :), view(G[r], :, :, gn))
                end
            end
        end
    end
    mx
end
=#
