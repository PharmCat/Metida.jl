################################################################################
#                         R MATRIX FUNCTIONS
################################################################################

################################################################################
#=
function rmat_base_inc_b!(mx, θ, zrv, covstr)
    rmat!(mx, θ, zrv, covstr.repeated.covtype.s)
end
=#
################################################################################
################################################################################
#=
function rmat_base_inc!(mx, θ, covstr, block, sblock)
    zblock    = view(covstr.rz, block, :)
    @simd for i ∈ axes(sblock[end], 1)
        @inbounds rmat_base_inc_b!(view(mx, sblock[end][i],  sblock[end][i]), θ, view(zblock,  sblock[end][i], :), covstr)
    end
    mx
end
=#
@noinline function rmat_base_inc!(mx, θ, covstr, bi)
    en        = covstr.rn + 1
    block     = covstr.vcovblock[bi]
    zblock    = view(covstr.rz, block, :)
    @simd for i = 1:subjn(covstr, en, bi)
        sb = getsubj(covstr, en, bi, i)
        rmat!(view(mx, sb, sb), θ, view(zblock, sb, :), covstr.repeated.covtype.s)
    end
    mx
end
################################################################################
function rmat!(::Any, ::Any, ::Any,  ::AbstractCovarianceType)
    error("No rmat! method defined for thit structure!")
end
#SI
Base.@propagate_inbounds function rmat!(mx, θ, ::AbstractMatrix, ::SI_)
    val = θ[1]^2
    @inbounds @simd for i ∈ axes(mx, 1)
            mx[i, i] += val
    end
    mx
end
#DIAG
function rmat!(mx, θ, rz, ::DIAG_)
    #=@turbo=#  for i ∈ axes(mx, 1)
        @inbounds @simd for c ∈ axes(θ, 1)
            mx[i, i] += rz[i, c] * θ[c] * θ[c]
        end
    end
    mx
end
#AR
function rmat!(mx, θ, ::AbstractMatrix, ::AR_)
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
    mx
end
#ARH
function rmat!(mx, θ, rz, ::ARH_)
    vec = tmul_unsafe(rz, θ)
    rn    = size(mx, 1)
    if rn > 1
        for m = 1:rn - 1
            @inbounds @simd for n = m + 1:rn
                 mx[m, n] += vec[m] * vec[n] * θ[end] ^ (n - m)
            end
        end
    end
    @inbounds  for m ∈ axes(mx, 1)
        mx[m, m] += vec[m] * vec[m]
    end
    mx
end
#CS
function rmat!(mx, θ, ::AbstractMatrix,  ::CS_)
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
    mx
end
#CSH
function rmat!(mx, θ, rz, ::CSH_)
    vec = tmul_unsafe(rz, θ)
    s    = size(mx, 1)
    if s > 1
        θend = last(θ)
        for n = 2:s
            @inbounds vecnθend = vec[n] * θend
            @inbounds @simd for m = 1:n-1
                mx[m, n] += vec[m] * vecnθend
            end
        end
    end
    @inbounds  for m ∈ axes(mx, 1)
        mx[m, m] += vec[m] * vec[m]
    end
    mx
end
################################################################################
#ARMA
function rmat!(mx, θ, ::AbstractMatrix,  ::ARMA_)
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
    mx
end
################################################################################
#TOEPP
function rmat!(mx, θ, ::AbstractMatrix, ct::TOEPP_)
    de  = θ[1] ^ 2    #diagonal element
    s   = size(mx, 1) #size
    @inbounds @simd for i = 1:s
        mx[i, i] += de
    end
    if s > 1 && ct.p > 1
        for m = 1:s - 1
            for n = m + 1:(m + ct.p - 1 > s ? s : m + ct.p - 1)
                @inbounds  mx[m, n] += de * θ[n - m + 1]
            end
        end
    end
    mx
end
################################################################################
#TOEPHP
function rmat!(mx, θ, rz, ct::TOEPHP_)
    l     = size(rz, 2)
    vec   = rz * (θ[1:l])
    s   = size(mx, 1) #size
    if s > 1 && ct.p > 1
        for m = 1:s - 1
            for n = m + 1:(m + ct.p - 1 > s ? s : m + ct.p - 1)
                @inbounds  mx[m, n] += vec[m] * vec[n] * θ[n - m + l]
            end
        end
    end
    @inbounds @simd for m = 1:s
        mx[m, m] += vec[m] * vec[m]
    end
    mx
end
################################################################################
#=
 Base.@propagate_inbounds function edistance(i::AbstractVector{T1}, j::AbstractVector{T2}) where T1 where T2
    length(i) == length(j) || error("length i not equal j")
    sum = zero(promote_type(T1, T2))
    for c = 1:length(i)
        sum += (i[c]-j[c])^2
    end
    return sqrt(sum)
end
=#
function edistance(mx::AbstractMatrix{T}, i::Int, j::Int) where T
    sum = zero(T)
    @inbounds for c = 1:size(mx, 2)
        sum += (mx[i, c] - mx[j, c])^2
    end
    return sqrt(sum)
end
################################################################################
#SPEXP
function rmat!(mx, θ, rz,  ::SPEXP_)
    σ²    = θ[1]^2
    #θe    = exp(θ[2])
    θe    = θ[2]
    θe    = abs(θe) < eps() ? sqrt(eps()) : abs(θe)
    #θe    = abs(θ[2])
    rn    = size(mx, 1)
    @simd for i = 1:size(mx, 1)
        @inbounds mx[i, i] += σ²
    end
    if rn > 1
        for m = 1:rn - 1
            @simd for n = m + 1:rn
                mx[m, n] += σ² * exp(-edistance(rz, m, n) / θe)
            end
        end
    end
    mx
end
################################################################################
#SPPOW
function rmat!(mx, θ, rz,  ::SPPOW_)
    σ²    = θ[1]^2
    ρ     = θ[2]
    rn    = size(mx, 1)
    @simd for i = 1:size(mx, 1)
        @inbounds mx[i, i] += σ²
    end
    if rn > 1
        for m = 1:rn - 1
            @simd for n = m + 1:rn
                ed = edistance(rz, m, n)
                mx[m, n] += ρ > 0 ? σ² * ρ^ed : σ² * ρ^ed * cos(π * ed)
                #mx[m, n] += ρ > 0 ? σ² * ρ^ed : σ² * abs(ρ)^ed * sign(ρ)
            end
        end
    end
    mx
end

#SPGAU
function rmat!(mx, θ, rz,  ::SPGAU_)
    σ²    = θ[1]^2
    #θe    = exp(θ[2])
    θe    = θ[2]
    θe    = abs(θe) < eps() ? sqrt(eps()) : θe^2
    #θe    = θ[2]
    #θe    = abs(θ[2])
    rn    = size(mx, 1)
    @simd for i = 1:size(mx, 1)
        @inbounds mx[i, i] += σ²
    end
    if rn > 1
        for m = 1:rn - 1
            @simd for n = m + 1:rn
                mx[m, n] += σ² * exp(- (edistance(rz, m, n)^2) / θe)
            end
        end
    end
    mx
end
################################################################################
#SPEXPD cos(pidij)
function rmat!(mx, θ, rz,  ::SPEXPD_)
    σ²    = θ[2]^2
    σ²d   = θ[1]^2 + σ²
    θe    = θ[3]
    θe    = iszero(θe) ? 1e-16 : abs(θe)
    rn    = size(mx, 1)
    @simd for i = 1:size(mx, 1)
        @inbounds mx[i, i] += σ²d
    end
    if rn > 1
        for m = 1:rn - 1
            @simd for n = m + 1:rn
                mx[m, n] += σ² * exp(-edistance(rz, m, n) / θe)
            end
        end
    end
    mx
end
#SPPOWD
function rmat!(mx, θ, rz,  ::SPPOWD_)
    σ²    = θ[2]^2
    σ²d   = θ[1]^2 + σ²
    ρ     = θ[3]
    rn    = size(mx, 1)
    @simd for i = 1:size(mx, 1)
        @inbounds mx[i, i] += σ²d
    end
    if rn > 1
        for m = 1:rn - 1
            @simd for n = m + 1:rn
                mx[m, n] += σ² * ρ^edistance(rz, m, n)
            end
        end
    end
    mx
end
#SPGAUD
function rmat!(mx, θ, rz,  ::SPGAUD_)
    σ²    = θ[2]^2
    σ²d   = θ[1]^2 + σ²
    θe    = θ[3]
    θe    = iszero(θe) ? 1e-16 : θe^2
    rn    = size(mx, 1)
    @simd for i = 1:size(mx, 1)
        @inbounds mx[i, i] += σ²d
    end
    if rn > 1
        for m = 1:rn - 1
            @simd for n = m + 1:rn
                mx[m, n] += σ² * exp(- (edistance(rz, m, n)^2) / θe)
            end
        end
    end
    mx
end

#UN
function unrmat(θ::AbstractVector{T}, rz) where T
    rm = size(rz, 2)
    mx = zeros(T, rm, rm)
    for i = 1:rm
        mx[i, i] = θ[i]
    end
    if rm > 1
        for m = 1:rm - 1
            @inbounds @simd for n = m + 1:rm
                mx[m, n] += mx[m, m] * mx[n, n] * θ[rm+tpnum(m, n, rm)]
            end
        end
    end
    @inbounds @simd for m = 1:rm
        mx[m, m] *= mx[m, m]
    end
    Symmetric(mx)
end
function rmat!(mx, θ, rz::AbstractMatrix, ::UN_)
    vec    = tmul_unsafe(rz, θ)
    rm     = size(mx, 1)
    rcov  = unrmat(θ, rz)
    mulαβαtinc!(mx, rz, rcov)
    mx
end
###############################################################################
###############################################################################
###############################################################################
###############################################################################