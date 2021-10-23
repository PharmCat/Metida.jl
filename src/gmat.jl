###############################################################################
#                       G MATRIX FUNCTIONS
################################################################################

################################################################################
function gmat_switch!(G, θ, covstr, r)
    gmat!(G, θ[covstr.tr[r]], covstr.random[r].covtype.s)
    G
end
################################################################################
function zgz_base_inc!(mx::AbstractArray{T}, θ::AbstractArray{T}, covstr, block, sblock) where T
    if covstr.random[1].covtype.z
        #length of random to covstr
        for r = 1:covstr.rn
            G = fill!(Symmetric(Matrix{T}(undef, covstr.q[r], covstr.q[r])), zero(T))
            gmat_switch!(G.data, θ, covstr, r)
            zblock    = view(covstr.z, block, covstr.zrndur[r])
            @inbounds for i = 1:length(sblock[r])
                mulαβαtinc!(view(mx, sblock[r][i], sblock[r][i]), view(zblock, sblock[r][i], :), G)
            end
        end
    end
    #    fill!(mx, zero(T))
    #end
    mx
end
################################################################################
#SI
function gmat!(mx, θ, ::SI_)
    val = θ[1] ^ 2
    @inbounds @simd for i = 1:size(mx, 1)
        mx[i, i] = val
    end
    nothing
end
#DIAG
function gmat!(mx, θ, ::DIAG_)
    @inbounds @simd for i = 1:size(mx, 1)
        mx[i, i] = θ[i] ^ 2
    end
    nothing
end
#function gmat_vc!(mx, θ::Vector{T}, ::Int, ::CovarianceType) where T
#    nothing
#end
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
    nothing
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
    nothing
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
    nothing
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
    nothing
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
    nothing
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
    nothing
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
    nothing
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
    nothing
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
    nothing
end
