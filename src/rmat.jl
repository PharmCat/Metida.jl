################################################################################
#                         R MATRIX FUNCTIONS
################################################################################

################################################################################
function rmat_base_inc_b!(mx, θ, zrv, covstr)
    rmat!(mx, θ, zrv, covstr.repeated.covtype.s)
end
################################################################################
################################################################################
function rmat_base_inc!(mx, θ, covstr, block, sblock)
    zblock    = view(covstr.rz, block, :)
    @simd for i ∈ axes(sblock[end], 1)
        @inbounds rmat_base_inc_b!(view(mx, sblock[end][i],  sblock[end][i]), θ, view(zblock,  sblock[end][i], :), covstr)
    end
    mx
end
################################################################################
#SI
function rmat!(mx, θ, ::AbstractMatrix, ::SI_)
    θsq = θ[1]*θ[1]
    @inbounds @simd for i ∈ axes(mx, 1)
            mx[i, i] += θsq
    end
    nothing
end
#DIAG
function rmat!(mx, θ, rz, ::DIAG_)
    #=@turbo=# @inbounds  for i ∈ axes(mx, 1)
        for c ∈ axes(θ, 1)
            mx[i, i] += rz[i, c] * θ[c] * θ[c]
        end
    end
    nothing
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
    nothing
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
    nothing
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
    nothing
end
#CSH
function rmat!(mx, θ, rz, ::CSH_)
    #vec   = rz * (θ[1:end-1])
    #=
    vec = zeros(T, size(rz, 1))
    #=@turbo=# @inbounds  for r ∈ axes(rz, 1)
        for i ∈ axes(rz, 2)
            vec[r] += rz[r, i] * θ[i]
        end
    end
    =#
    vec = tmul_unsafe(rz, θ)
    rn    = size(mx, 1)
    if rn > 1
        for m = 1:rn - 1
            @inbounds @simd for n = m + 1:rn
                mx[m, n] += vec[m] * vec[n] * θ[end]
            end
        end
    end
    #=@turbo=# @inbounds  for m ∈ axes(mx, 1)
        mx[m, m] += vec[m] * vec[m]
    end
    nothing
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
    nothing
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
    nothing
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
    nothing
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
    θe    = exp(θ[2])
    #θe    = θ[2]
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
    nothing
end
