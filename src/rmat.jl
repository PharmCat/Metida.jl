################################################################################
#                         R MATRIX FUNCTIONS
################################################################################

################################################################################
function rmat_base_inc_b!(mx::AbstractMatrix{T}, θ::AbstractVector{T}, zrv, covstr::CovStructure{T2}) where T where T2
        if covstr.repeated.covtype.s == :SI
            rmatp_si!(mx, θ, zrv, covstr.repeated.covtype.p)
        elseif covstr.repeated.covtype.s == :DIAG
            rmatp_diag!(mx, θ, zrv, covstr.repeated.covtype.p)
        elseif covstr.repeated.covtype.s == :AR
            rmatp_ar!(mx, θ, zrv, covstr.repeated.covtype.p)
        elseif covstr.repeated.covtype.s == :ARH
            rmatp_arh!(mx, θ, zrv, covstr.repeated.covtype.p)
        elseif covstr.repeated.covtype.s == :CSH
            rmatp_csh!(mx, θ, zrv, covstr.repeated.covtype.p)
        elseif covstr.repeated.covtype.s == :CS
            rmatp_cs!(mx, θ, zrv, covstr.repeated.covtype.p)
        elseif covstr.repeated.covtype.s == :ARMA
            rmatp_arma!(mx, θ, zrv, covstr.repeated.covtype.p)
        elseif covstr.repeated.covtype.s == :TOEPP
            rmatp_toepp!(mx, θ, zrv, covstr.repeated.covtype.p)
        elseif covstr.repeated.covtype.s == :TOEPHP
            rmatp_toephp!(mx, θ, zrv, covstr.repeated.covtype.p)
        elseif covstr.repeated.covtype.s == :FUNC
            covstr.repeated.covtype.p.xmat!(mx, θ, zrv, covstr.repeated.covtype.p)
        end
end
################################################################################
################################################################################
function rmat_base_inc!(mx, θ::AbstractVector{T}, covstr, block, sblock) where T
    zblock    = view(covstr.rz, block, :)
    for i = 1:length(sblock[end])
        #unsafe array uview() ?
        rmat_base_inc_b!(view(mx, sblock[end][i],  sblock[end][i]), θ, view(zblock,  sblock[end][i], :), covstr)
    end
    mx
end
################################################################################
function rmatp_si!(mx::AbstractMatrix{T}, θ::Vector{T}, ::AbstractMatrix, p) where T
    θsq = θ[1]*θ[1]
    for i = 1:size(mx, 1)
            mx[i, i] += θsq
    end
    nothing
end
function rmatp_diag!(mx::AbstractMatrix{T}, θ::Vector{T}, rz, p) where T
    for i = 1:size(mx, 1)
        for c = 1:length(θ)
            mx[i, i] += rz[i, c] * θ[c] * θ[c]
        end
    end
    nothing
end
function rmatp_ar!(mx, θ::Vector{T}, ::AbstractMatrix, p) where T
    rn  = size(mx, 1)
    de  = θ[1] ^ 2
    for m = 1:rn
        mx[m, m] += de
    end
    if rn > 1
        for m = 1:rn - 1
            for n = m + 1:rn
                ode = de * θ[2] ^ (n - m)
                @inbounds mx[m, n] += ode
                #@inbounds mx[n, m] = mx[m, n]
            end
        end
    end
    nothing
end
function rmatp_arh!(mx, θ::Vector{T}, rz, p) where T
    vec   = rz * (θ[1:end-1])
    rn    = size(mx, 1)
    if rn > 1
        for m = 1:rn - 1
            for n = m + 1:rn
                @inbounds mx[m, n] += vec[m] * vec[n] * θ[end] ^ (n - m)
                #@inbounds mx[n, m] = mx[m, n]
            end
        end
    end
    for m = 1:rn
        @inbounds mx[m, m] += vec[m] * vec[m]
    end
    nothing
end
function rmatp_cs!(mx, θ::Vector{T}, ::AbstractMatrix, p) where T
    rn    = size(mx, 1)
    θsq   =  θ[1]*θ[1]
    θsqp  =  θsq*θ[2]
    for i = 1:size(mx, 1)
        mx[i, i] += θsq
    end
    if rn > 1
        for m = 1:rn - 1
            for n = m + 1:rn
                @inbounds mx[m, n] += θsqp
                #@inbounds mx[n, m] = mx[m, n]
            end
        end
    end
    nothing
end
function rmatp_csh!(mx, θ::Vector{T}, rz, p) where T
    vec   = rz * (θ[1:end-1])
    rn    = size(mx, 1)
    if rn > 1
        for m = 1:rn - 1
            for n = m + 1:rn
                @inbounds mx[m, n] += vec[m] * vec[n] * θ[end]
                #@inbounds mx[n, m] = mx[m, n]
            end
        end
    end
    for m = 1:rn
        @inbounds mx[m, m] += vec[m] * vec[m]
    end
    nothing
end
################################################################################
function rmatp_arma!(mx, θ::Vector{T}, ::AbstractMatrix, p) where T
    rn  = size(mx, 1)
    de  = θ[1] ^ 2
    for m = 1:rn
        mx[m, m] += de
    end
    if rn > 1
        for m = 1:rn - 1
            for n = m + 1:rn
                ode = de * θ[2] * θ[3] ^ (n - m - 1)
                @inbounds mx[m, n] += ode
                #@inbounds mx[n, m] = mx[m, n]
            end
        end
    end
    nothing
end
################################################################################
function rmatp_toepp!(mx, θ::Vector{T}, ::AbstractMatrix, p) where T
    de  = θ[1] ^ 2    #diagonal element
    s   = size(mx, 1) #size
    for i = 1:s
        mx[i, i] += de
    end
    if s > 1 && p > 1
        for m = 1:s - 1
            for n = m + 1:(m + p - 1 > s ? s : m + p - 1)
                ode = de * θ[n - m + 1]
                @inbounds mx[m, n] += ode
                #mx[n, m] = mx[m, n]
            end
        end
    end
    nothing
end
################################################################################
function rmatp_toephp!(mx, θ::Vector{T}, rz, p) where T
    l     = size(rz, 2)
    vec   = rz * (θ[1:l])
    s   = size(mx, 1) #size
    if s > 1 && p > 1
        for m = 1:s - 1
            for n = m + 1:(m + p - 1 > s ? s : m + p - 1)
                @inbounds mx[m, n] += vec[m] * vec[n] * θ[n - m + l]
                #mx[n, m] = mx[m, n]
            end
        end
    end
    for m = 1:s
        @inbounds mx[m, m] += vec[m] * vec[m]
    end
    nothing
end
