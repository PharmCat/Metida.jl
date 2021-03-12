###############################################################################
#                       G MATRIX FUNCTIONS
################################################################################
#=
function gmat_base(θ::Vector{T}, covstr) where T
    q = size(covstr.z, 2)
    mx = zeros(T, q, q)
    if covstr.random[1].covtype.s != :ZERO
        for i = 1:length(covstr.random)
            vmxr = (1 + sum(covstr.q[1:i]) - covstr.q[i]):sum(covstr.q[1:i])
            vmx = view(mx, vmxr, vmxr)
            gmat_switch!(vmx, θ, covstr, i)
        end
    end
    mx
end
=#
################################################################################
function gmat_switch!(G, θ, covstr, r)
    if covstr.random[r].covtype.s == :SI
        gmat_si!(G, θ[covstr.tr[r]],  covstr.random[r].covtype.p) # i > r
    elseif covstr.random[r].covtype.s == :DIAG
        gmat_diag!(G, θ[covstr.tr[r]],  covstr.random[r].covtype.p)
    elseif covstr.random[r].covtype.s == :AR
        gmat_ar!(G, θ[covstr.tr[r]],  covstr.random[r].covtype.p)
    elseif covstr.random[r].covtype.s == :ARH
        gmat_arh!(G, θ[covstr.tr[r]],   covstr.random[r].covtype.p)
    elseif covstr.random[r].covtype.s == :CSH
        gmat_csh!(G, θ[covstr.tr[r]],  covstr.random[r].covtype.p)
    elseif covstr.random[r].covtype.s == :CS
        gmat_cs!(G, θ[covstr.tr[r]],   covstr.random[r].covtype.p)
    elseif covstr.random[r].covtype.s == :ARMA
        gmat_arma!(G, θ[covstr.tr[r]],  covstr.random[r].covtype.p)
    elseif covstr.random[r].covtype.s == :TOEP
        gmat_toep!(G, θ[covstr.tr[r]],  covstr.random[r].covtype.p)
    elseif covstr.random[r].covtype.s == :TOEPP
        gmat_toepp!(G, θ[covstr.tr[r]],  covstr.random[r].covtype.p)
    elseif covstr.random[r].covtype.s == :TOEPH
        gmat_toeph!(G, θ[covstr.tr[r]],  covstr.random[r].covtype.p)
    elseif covstr.random[r].covtype.s == :TOEPHP
        gmat_toephp!(G, θ[covstr.tr[r]],  covstr.random[r].covtype.p)
    elseif covstr.random[r].covtype.s == :FUNC
         covstr.random[r].covtype.p.xmat!(G, θ[covstr.tr[r]], covstr.random[r].covtype.p)
    end
    G
end
################################################################################
function zgz_base_inc!(mx, θ::Vector{T}, covstr, block, sblock) where T
    #q = sum(length.(covstr.block[1]))
    #q = 0
    #for i in covstr.block[1]
    #    q += length(i)
    #end
    if covstr.random[1].covtype.s != :ZERO
        #length of random to covstr
        for r = 1:covstr.rn
            G = zeros(T, covstr.q[r], covstr.q[r])
            gmat_switch!(G, θ, covstr, r)
            zblock    = view(covstr.z, block, covstr.zrndur[r])
            for i = 1:length(sblock[r])
                #unsafe view?
                mulαβαtinc!(view(mx, sblock[r][i], sblock[r][i]), view(zblock, sblock[r][i], :), Symmetric(G))
            end
        end
    end
    mx
end
################################################################################
function gmat_si!(mx, θ::Vector{T}, p) where T
    val = θ[1] ^ 2
    for i = 1:size(mx, 1)
        mx[i, i] = val
    end
    nothing
end
function gmat_diag!(mx, θ::Vector{T}, p) where T
    for i = 1:size(mx, 1)
        mx[i, i] = θ[i] ^ 2
    end
    nothing
end
#function gmat_vc!(mx, θ::Vector{T}, ::Int, ::CovarianceType) where T
#    nothing
#end
function gmat_ar!(mx, θ::Vector{T}, p) where T
    de  = θ[1] ^ 2
    s   = size(mx, 1)
    for i = 1:s
        mx[i, i] = de
    end
    if s > 1
        for m = 1:s - 1
            for n = m + 1:s
                ode = de * θ[2] ^ (n - m)
                @inbounds mx[m, n] = ode
                #@inbounds mx[n, m] = mx[m, n]
            end
        end
    end
    nothing
end
function gmat_arh!(mx, θ::Vector{T}, p) where T
    s = size(mx, 1)
    for m = 1:s
        @inbounds mx[m, m] = θ[m]
    end
    if s > 1
        for m = 1:s - 1
            for n = m + 1:s
                @inbounds mx[m, n] = mx[m, m] * mx[n, n] * θ[end] ^ (n - m)
                #@inbounds mx[n, m] = mx[m, n]
            end
        end
    end
    @simd for m = 1:s
        @inbounds mx[m, m] = mx[m, m] * mx[m, m]
    end
    nothing
end
function gmat_cs!(mx, θ::Vector{T}, p) where T
    s = size(mx, 1)
    mx .= θ[1]^2
    if s > 1
        for m = 1:s - 1
            for n = m + 1:s
                @inbounds mx[m, n] = mx[m, m] * θ[2]
                #@inbounds mx[n, m] = mx[m, n]
            end
        end
    end
    nothing
end
function gmat_csh!(mx::AbstractMatrix{T}, θ::Vector{T}, p) where T
    s = size(mx, 1)
    for m = 1:s
        @inbounds mx[m, m] = θ[m]
    end
    if s > 1
        for m = 1:s - 1
            for n = m + 1:s
                @inbounds mx[m, n] = mx[m, m] * mx[n, n] * θ[end]
                #@inbounds mx[n, m] = mx[m, n]
            end
        end
    end
    @simd for m = 1:s
        @inbounds mx[m, m] = mx[m, m] * mx[m, m]
    end
    nothing
end
################################################################################
function gmat_arma!(mx, θ::Vector{T}, p) where T
    de  = θ[1] ^ 2
    s   = size(mx, 1)
    for i = 1:s
        mx[i, i] = de
    end
    if s > 1
        for m = 1:s - 1
            for n = m + 1:s
                ode = de * θ[2] * θ[3] ^ (n - m - 1)
                @inbounds mx[m, n] = ode
                #@inbounds mx[n, m] = mx[m, n]
            end
        end
    end
    nothing
end
function gmat_toep!(mx, θ::Vector{T}, p) where T
    de  = θ[1] ^ 2    #diagonal element
    s   = size(mx, 1) #size
    for i = 1:s
        mx[i, i] = de
    end
    if s > 1
        for m = 1:s - 1
            for n = m + 1:s
                @inbounds ode = de * θ[n-m+1]
                @inbounds mx[m, n] = ode
                #@inbounds mx[n, m] = mx[m, n]
            end
        end
    end
    nothing
end
function gmat_toepp!(mx, θ::Vector{T}, p) where T
    de  = θ[1] ^ 2    #diagonal element
    s   = size(mx, 1) #size
    for i = 1:s
        mx[i, i] = de
    end
    if s > 1 && p > 1
        for m = 1:s - 1
            for n = m + 1:(m + p - 1 > s ? s : m + p - 1)
                @inbounds ode = de * θ[n - m + 1]
                @inbounds mx[m, n] = ode
                #@inbounds mx[n, m] = mx[m, n]
            end
        end
    end
    nothing
end
function gmat_toeph!(mx, θ::Vector{T}, p) where T
    s = size(mx, 2)
    for m = 1:s
        @inbounds mx[m, m] = θ[m]
    end
    if s > 1
        for m = 1:s - 1
            for n = m + 1:s
                @inbounds mx[m, n] = mx[m, m] * mx[n, n] * θ[n-m+s]
                #@inbounds mx[n, m] = mx[m, n]
            end
        end
    end
    @simd for m = 1:s
        @inbounds mx[m, m] = mx[m, m] * mx[m, m]
    end
    nothing
end
function gmat_toephp!(mx, θ::Vector{T}, p) where T
    s = size(mx, 2)
    for m = 1:s
        @inbounds mx[m, m] = θ[m]
    end
    if s > 1 && p > 1
        for m = 1:s - 1
            for n = m + 1:(m + p - 1 > s ? s : m + p - 1)
                @inbounds mx[m, n] = mx[m, m] * mx[n, n] * θ[n - m + s]
                #@inbounds mx[n, m] = mx[m, n]
            end
        end
    end
    @simd for m = 1:s
        @inbounds mx[m, m] = mx[m, m] * mx[m, m]
    end
    nothing
end
