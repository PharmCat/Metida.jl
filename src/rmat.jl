################################################################################
#                         R MATRIX FUNCTIONS
################################################################################
function rmat_basep!(mx, θ::AbstractVector{T}, zrv, covstr) where T
    if covstr.repeated.covtype.s == :SI
        rmatp_si!(mx, θ, zrv, covstr.repeated.covtype)
    elseif covstr.repeated.covtype.s == :VC
        rmatp_vc!(mx, θ, zrv, covstr.repeated.covtype)
    elseif covstr.repeated.covtype.s == :AR
        rmatp_ar!(mx, θ, zrv, covstr.repeated.covtype)
    elseif covstr.repeated.covtype.s == :ARH
        rmatp_arh!(mx, θ, zrv, covstr.repeated.covtype)
    elseif covstr.repeated.covtype.s == :CSH
        rmatp_csh!(mx, θ, zrv, covstr.repeated.covtype)
    elseif covstr.repeated.covtype.s == :CS
        rmatp_cs!(mx, θ, zrv, covstr.repeated.covtype)
    else
        throw(ErrorException("Unknown covariance structure: $(covstr.repeated.covtype.s)"))
    end
end
function rmatp_si!(mx, θ::Vector{T}, ::AbstractMatrix, ::CovarianceType) where T
    θsq = θ[1]*θ[1]
    for i = 1:size(mx, 1)
            mx[i, i] += θsq
    end
    nothing
end
function rmatp_vc!(mx, θ::Vector{T}, rz,  ::CovarianceType) where T
    for i = 1:size(mx, 1)
        for c = 1:length(θ)
            mx[i, i] += θ[c]*θ[c]*rz[i, c]
        end
    end
    nothing
end
function rmatp_ar!(mx, θ::Vector{T}, rz, ::CovarianceType) where T
    rn  = size(mx, 1)
    mx  = Matrix{T}(undef, rn, rn)
    de  = θ[1] ^ 2
    for m = 1:rn
        mx[m, m] += de
    end
    if rn > 1
        for m = 1:rn - 1
            for n = m + 1:rn
                ode = de * θ[2] ^ (n - m)
                @inbounds mx[m, n] += ode
                @inbounds mx[n, m] = mx[m, n]
            end
        end
    end
    nothing
end
function rmatp_arh!(mx, θ::Vector{T}, rz, ::CovarianceType) where T
    vec   = rz * (θ[1:end-1])
    rn    = size(mx, 1)
    if rn > 1
        for m = 1:rn - 1
            for n = m + 1:rn
                @inbounds mx[m, n] += vec[m] * vec[n] * θ[end] ^ (n - m)
                @inbounds mx[n, m] = mx[m, n]
            end
        end
    end
    for m = 1:rn
        @inbounds mx[m, m] += vec[m] * vec[m]
    end
    nothing
end
function rmatp_cs!(mx, θ::Vector{T}, ::AbstractMatrix,  ::CovarianceType) where T
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
                @inbounds mx[n, m] = mx[m, n]
            end
        end
    end
    nothing
end
function rmatp_csh!(mx, θ::Vector{T}, rz, ::CovarianceType) where T
    vec   = rz * (θ[1:end-1])
    rn    = size(mx, 1)
    if rn > 1
        for m = 1:rn - 1
            for n = m + 1:rn
                @inbounds mx[m, n] += vec[m] * vec[n] * θ[end]
                @inbounds mx[n, m] = mx[m, n]
            end
        end
    end
    for m = 1:rn
        @inbounds mx[m, m] += vec[m] * vec[m]
    end
    nothing
end