###############################################################################
#                       G MATRIX FUNCTIONS
################################################################################
function gmat_base(θ::Vector{T}, covstr) where T
    q = size(covstr.z, 2)
    mx = zeros(T, q, q)
    for i = 1:length(covstr.random)
        vmxr = (1 + sum(covstr.q[1:i]) - covstr.q[i]):sum(covstr.q[1:i])
        vmx = view(mx, vmxr, vmxr)
        gmat_switch!(vmx, θ, covstr, i)
    end
    mx
end
################################################################################
function gmat_switch!(G, θ, covstr, i)
    if covstr.random[i].covtype.s == :SI
        gmat_si!(G, θ[covstr.tr[i]], covstr.q[i], covstr.random[i].covtype) # i > r
    elseif covstr.random[i].covtype.s == :VC
        gmat_vc!(G, θ[covstr.tr[i]], covstr.q[i], covstr.random[i].covtype)
    elseif covstr.random[i].covtype.s == :AR
        gmat_ar!(G, θ[covstr.tr[i]], covstr.q[i], covstr.random[i].covtype)
    elseif covstr.random[i].covtype.s == :ARH
        gmat_arh!(G, θ[covstr.tr[i]], covstr.q[i], covstr.random[i].covtype)
    elseif covstr.random[i].covtype.s == :CSH
        gmat_csh!(G, θ[covstr.tr[i]], covstr.q[i], covstr.random[i].covtype)
    elseif covstr.random[i].covtype.s == :CS
        gmat_cs!(G, θ[covstr.tr[i]], covstr.q[i], covstr.random[i].covtype)
    else
        throw(ErrorException("Unknown covariance structure: $(covstr.random[i].covtype.s), n = $(i)"))
    end
    G
end
################################################################################
function gmat_base_z(θ::Vector{T}, covstr) where T
    q = sum(length.(covstr.block[1]))
    mx = zeros(T, q, q)
    for r = 1:length(covstr.random)
        G = zeros(T, covstr.q[r], covstr.q[r])
        gmat_switch!(G, θ, covstr, r)
        for i = 1:length(covstr.block[r])
            mulαβαtinc!(view(mx, covstr.block[r][i], covstr.block[r][i]), view(covstr.z, covstr.block[r][i], covstr.zr[r]), G)
        end
    end
    mx
end
################################################################################
function gmat_si!(mx, θ::Vector{T}, zn::Int, ::CovarianceType) where T
    val = θ[1] ^ 2
    for i = 1:size(mx, 1)
        mx[i, i] = val
    end
    nothing
end
function gmat_vc!(mx, θ::Vector{T}, ::Int, ::CovarianceType) where T
    for i = 1:size(mx, 1)
        mx[i, i] = θ[i] ^ 2
    end
    nothing
end
function gmat_ar!(mx, θ::Vector{T}, zn::Int, ::CovarianceType) where T
    mx .= θ[1] ^ 2
    if zn > 1
        for m = 1:zn - 1
            for n = m + 1:zn
                @inbounds mx[m, n] = mx[m, m] * θ[2] ^ (n - m)
                @inbounds mx[n, m] = mx[m, n]
            end
        end
    end
    nothing
end
function gmat_arh!(mx, θ::Vector{T}, ::Int, ::CovarianceType) where T
    s = size(mx, 1)
    for m = 1:s
        @inbounds mx[m, m] = θ[m]
    end
    if s > 1
        for m = 1:s - 1
            for n = m + 1:s
                @inbounds mx[m, n] = mx[m, m] * mx[n, n] * θ[end] ^ (n - m)
                @inbounds mx[n, m] = mx[m, n]
            end
        end
    end
    for m = 1:s
        @inbounds mx[m, m] = mx[m, m] * mx[m, m]
    end
    nothing
end
function gmat_cs!(mx, θ::Vector{T}, ::Int, ::CovarianceType) where T
    s = size(mx, 1)
    mx .= θ[1]^2
    if s > 1
        for m = 1:s - 1
            for n = m + 1:s
                @inbounds mx[m, n] = mx[m, m] * θ[2]
                @inbounds mx[n, m] = mx[m, n]
            end
        end
    end
    nothing
end
function gmat_csh!(mx, θ::Vector{T}, ::Int, ::CovarianceType) where T
    s = size(mx, 1)
    for m = 1:s
        @inbounds mx[m, m] = θ[m]
    end
    if s > 1
        for m = 1:s - 1
            for n = m + 1:s
                @inbounds mx[m, n] = mx[m, m] * mx[n, n] * θ[end]
                @inbounds mx[n, m] = mx[m, n]
            end
        end
    end
    for m = 1:s
        @inbounds mx[m, m] = mx[m, m] * mx[m, m]
    end
    nothing
end
################################################################################
