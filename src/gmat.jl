###############################################################################
#                       G MATRIX FUNCTIONS
################################################################################
function gmat_switch!(G, θ, covstr, r)
    if covstr.random[r].covtype.s == :SI
        gmat_si!(G, θ[covstr.tr[r]]) # i > r
    elseif covstr.random[r].covtype.s == :DIAG
        gmat_diag!(G, θ[covstr.tr[r]])
    elseif covstr.random[r].covtype.s == :AR
        gmat_ar!(G, θ[covstr.tr[r]])
    elseif covstr.random[r].covtype.s == :ARH
        gmat_arh!(G, θ[covstr.tr[r]])
    elseif covstr.random[r].covtype.s == :CSH
        gmat_csh!(G, θ[covstr.tr[r]])
    elseif covstr.random[r].covtype.s == :CS
        gmat_cs!(G, θ[covstr.tr[r]])
    elseif covstr.random[r].covtype.s == :ARMA
        gmat_arma!(G, θ[covstr.tr[r]])
    elseif covstr.random[r].covtype.s == :TOEP
        gmat_toep!(G, θ[covstr.tr[r]])
    elseif covstr.random[r].covtype.s == :TOEPP
        gmat_toepp!(G, θ[covstr.tr[r]],  covstr.random[r].covtype.p)
    elseif covstr.random[r].covtype.s == :TOEPH
        gmat_toeph!(G, θ[covstr.tr[r]])
    elseif covstr.random[r].covtype.s == :TOEPHP
        gmat_toephp!(G, θ[covstr.tr[r]],  covstr.random[r].covtype.p)
    elseif covstr.random[r].covtype.s == :FUNC
         covstr.random[r].covtype.f.xmat!(G, θ[covstr.tr[r]], covstr.random[r].covtype.p)
    end
    G
end
################################################################################
@inline function zgz_base_inc!(mx::AbstractArray{T}, θ::AbstractArray{T}, covstr, block, sblock) where T
    if covstr.random[1].covtype.s != :ZERO
        #length of random to covstr
        for r = 1:covstr.rn
            G = fill!(Symmetric(Matrix{T}(undef, covstr.q[r], covstr.q[r])), zero(T))
            gmat_switch!(G.data, θ, covstr, r)
            zblock    = view(covstr.z, block, covstr.zrndur[r])
            for i = 1:length(sblock[r])
                @inbounds mulαβαtinc!(view(mx, sblock[r][i], sblock[r][i]), view(zblock, sblock[r][i], :), G)
            end
        end
    end
    #    fill!(mx, zero(T))
    #end
    mx
end
################################################################################
function gmat_si!(mx, θ)
    val = θ[1] ^ 2
    @inbounds @simd for i = 1:size(mx, 1)
        mx[i, i] = val
    end
    nothing
end
function gmat_diag!(mx, θ)
    @inbounds @simd for i = 1:size(mx, 1)
        mx[i, i] = θ[i] ^ 2
    end
    nothing
end
#function gmat_vc!(mx, θ::Vector{T}, ::Int, ::CovarianceType) where T
#    nothing
#end
function gmat_ar!(mx, θ)
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
function gmat_arh!(mx, θ)
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
function gmat_cs!(mx, θ)
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
function gmat_csh!(mx, θ)
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
function gmat_arma!(mx, θ)
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
function gmat_toep!(mx, θ)
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
function gmat_toepp!(mx, θ, p::Int)
    de  = θ[1] ^ 2    #diagonal element
    s   = size(mx, 1) #size
    @inbounds @simd for i = 1:s
        mx[i, i] = de
    end
    if s > 1 && p > 1
        for m = 1:s - 1
            @inbounds @simd for n = m + 1:(m + p - 1 > s ? s : m + p - 1)
                mx[m, n] = de * θ[n - m + 1]
            end
        end
    end
    nothing
end
function gmat_toeph!(mx, θ)
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
function gmat_toephp!(mx, θ, p::Int)
    s = size(mx, 2)
    @inbounds @simd for m = 1:s
        mx[m, m] = θ[m]
    end
    if s > 1 && p > 1
        for m = 1:s - 1
            for n = m + 1:(m + p - 1 > s ? s : m + p - 1)
                @inbounds mx[m, n] = mx[m, m] * mx[n, n] * θ[n - m + s]
            end
        end
    end
    @inbounds @simd for m = 1:s
        mx[m, m] = mx[m, m] * mx[m, m]
    end
    nothing
end
