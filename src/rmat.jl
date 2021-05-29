################################################################################
#                         R MATRIX FUNCTIONS
################################################################################

################################################################################
function rmat_base_inc_b!(mx, θ, zrv, covstr)
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
        elseif covstr.repeated.covtype.s == :SPEXP
            rmatp_spexp!(mx, θ, zrv, covstr.repeated.covtype.p)
        end
end
################################################################################
################################################################################
function rmat_base_inc!(mx, θ, covstr, block, sblock)
    zblock    = view(covstr.rz, block, :)
    for i = 1:length(sblock[end])
        rmat_base_inc_b!(view(mx, sblock[end][i],  sblock[end][i]), θ, view(zblock,  sblock[end][i], :), covstr)
    end
    mx
end
################################################################################
function rmatp_si!(mx, θ, ::AbstractMatrix, p)
    θsq = θ[1]*θ[1]
    @inbounds @simd for i = 1:size(mx, 1)
            mx[i, i] += θsq
    end
    nothing
end
function rmatp_diag!(mx, θ, rz, p)
    for i = 1:size(mx, 1)
        @inbounds @simd for c = 1:length(θ)
            mx[i, i] += rz[i, c] * θ[c] * θ[c]
        end
    end
    nothing
end
function rmatp_ar!(mx, θ, ::AbstractMatrix, p)
    rn  = size(mx, 1)
    de  = θ[1] ^ 2
    @inbounds @simd for m = 1:rn
        mx[m, m] += de
    end
    if rn > 1
        for m = 1:rn - 1
            @inbounds @simd for n = m + 1:rn
                mx[m, n] += de * θ[2] ^ (n - m)
            end
        end
    end
    nothing
end
function rmatp_arh!(mx, θ, rz, p)
    vec   = rz * (θ[1:end-1])
    rn    = size(mx, 1)
    if rn > 1
        for m = 1:rn - 1
            @inbounds @simd for n = m + 1:rn
                 mx[m, n] += vec[m] * vec[n] * θ[end] ^ (n - m)
            end
        end
    end
    @inbounds @simd for m = 1:rn
        mx[m, m] += vec[m] * vec[m]
    end
    nothing
end
function rmatp_cs!(mx, θ, ::AbstractMatrix, p)
    rn    = size(mx, 1)
    θsq   =  θ[1]*θ[1]
    θsqp  =  θsq*θ[2]
    @inbounds @simd for i = 1:size(mx, 1)
        mx[i, i] += θsq
    end
    if rn > 1
        for m = 1:rn - 1
            @inbounds @simd for n = m + 1:rn
                mx[m, n] += θsqp
            end
        end
    end
    nothing
end
function rmatp_csh!(mx, θ, rz, p)
    vec   = rz * (θ[1:end-1])
    rn    = size(mx, 1)
    if rn > 1
        for m = 1:rn - 1
            @inbounds @simd for n = m + 1:rn
                mx[m, n] += vec[m] * vec[n] * θ[end]
            end
        end
    end
    @inbounds @simd for m = 1:rn
        mx[m, m] += vec[m] * vec[m]
    end
    nothing
end
################################################################################
function rmatp_arma!(mx, θ, ::AbstractMatrix, p)
    rn  = size(mx, 1)
    de  = θ[1] ^ 2
    @inbounds @simd for m = 1:rn
        mx[m, m] += de
    end
    if rn > 1
        for m = 1:rn - 1
            @inbounds @simd for n = m + 1:rn
                mx[m, n] += de * θ[2] * θ[3] ^ (n - m - 1)
            end
        end
    end
    nothing
end
################################################################################
function rmatp_toepp!(mx, θ, ::AbstractMatrix, p)
    de  = θ[1] ^ 2    #diagonal element
    s   = size(mx, 1) #size
    @inbounds @simd for i = 1:s
        mx[i, i] += de
    end
    if s > 1 && p > 1
        for m = 1:s - 1
            for n = m + 1:(m + p - 1 > s ? s : m + p - 1)
                @inbounds  mx[m, n] += de * θ[n - m + 1]
            end
        end
    end
    nothing
end
################################################################################
function rmatp_toephp!(mx, θ, rz, p)
    l     = size(rz, 2)
    vec   = rz * (θ[1:l])
    s   = size(mx, 1) #size
    if s > 1 && p > 1
        for m = 1:s - 1
            for n = m + 1:(m + p - 1 > s ? s : m + p - 1)
                @inbounds  mx[m, n] += vec[m] * vec[n] * θ[n - m + l]
            end
        end
    end
    @inbounds @simd for m = 1:s
        mx[m, m] += vec[m] * vec[m]
    end
    nothing
end
################################################################################
function edistance(i::AbstractVector{T1}, j::AbstractVector{T2}) where T1 where T2
    sum = zero(promote_type(T1, T2))
    for c = 1:length(i)
        sum += (i[c]-j[c])^2
    end
    return sqrt(sum)
end
function edistance(mx::AbstractMatrix{T}, i::Int, j::Int) where T
    sum = zero(T)
    for c = 1:size(mx, 2)
        sum += (mx[i,c] - mx[j,c])^2
    end
    return sqrt(sum)
end
################################################################################
function rmatp_spexp!(mx, θ, rz, p)
    σ²    = θ[1]^2
    θ     = exp(θ[2])
    rn    = size(mx, 1)
    @inbounds @simd for i = 1:size(mx, 1)
        mx[i, i] += σ²
    end
    if rn > 1
        for m = 1:rn - 1
            @inbounds @simd for n = m + 1:rn
                mx[m, n] += σ² * exp(-edistance(rz, m, n) / θ)
            end
        end
    end
    nothing
end
