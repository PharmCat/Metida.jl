#reml.jl

"""
    -2 log Restricted Maximum Likelihood;
"""
function reml_sweep(lmm, β, θ::Vector{T})::T where T <: Number
    n  = length(lmm.data.yv)
    N  = sum(length.(lmm.data.yv))
    G  = gmat_base(θ, lmm.covstr)
    c  = (N - lmm.rankx)*log(2π)
    θ₁ = zero(eltype(θ))
    θ₂ = zero(eltype(θ))
    θ₃ = zero(eltype(θ))
    θ2m  = zeros(eltype(θ), lmm.rankx, lmm.rankx)
    @inbounds for i = 1:n
        q   = length(lmm.data.yv[i])
        Vp  = mulαβαt3(lmm.data.zv[i], G, lmm.data.xv[i])
        V   = view(Vp, 1:q, 1:q)
        rmat_basep!(V, θ[lmm.covstr.tr[end]], lmm.data.zrv[i], lmm.covstr)

        θ₁  += logdet(V)
        sweep!(Vp, 1:q)
        iV  = Symmetric(utriaply!(x -> -x, Vp[1:q, 1:q]))
        mulαtβαinc!(θ2m, lmm.data.xv[i], iV)
        θ₃  += -Vp[end, end]
    end
    θ₂       = logdet(θ2m)
    return   -(θ₁ + θ₂ + θ₃ + c)
end

"""
    -2 log Restricted Maximum Likelihood; β calculation inside
"""
function reml_sweep_β(lmm, f::Function, θ::Vector{T}) where T <: Number
    f(θ)
    reml_sweep_β(lmm, θ)
end
"""
    -2 log Restricted Maximum Likelihood; β calculation inside
"""
function reml_sweep_β(lmm::LMM{T2}, θ::Vector{T})::Tuple{T, Vector{T}, Matrix{T}} where T <: Number where T2 <: Number
    n::Int        = length(lmm.data.block)
    N::Int        = length(lmm.data.yv)
    G::Matrix{T}  = gmat_base(θ, lmm.covstr)
    c::Float64    = (N - lmm.rankx)*log(2π)
    #---------------------------------------------------------------------------
    V⁻¹           = Vector{Matrix{T}}(undef, n)
    # Vector log determinant of V matrix
    θ₁::T         = zero(T)
    θ₂::Matrix{T} = zeros(T, lmm.rankx, lmm.rankx)
    θ₃::T         = zero(T)
    βm::Vector{T} = zeros(T, lmm.rankx)
    β::Vector{T}  = zeros(T, lmm.rankx)

    local q::Int
    local qswm::Int
    local R::Matrix{T}
    local Vp::Matrix{T}
    local logdetθ₂::T

    @inbounds for i = 1:n
        q   = length(lmm.data.block[i])
        Vp  = mulαβαt3(view(lmm.covstr.z, lmm.data.block[i],:), G, view(lmm.data.xv, lmm.data.block[i],:))
        V   = view(Vp, 1:q, 1:q)
        rmat_basep!(V, θ[lmm.covstr.tr[end]], view(lmm.covstr.rz, lmm.data.block[i],:), lmm.covstr)
        #θ₁  += logdet(V)

        θ₁  += logdet(V)

        sweep!(Vp, 1:q)
        V⁻¹[i] = Symmetric(utriaply!(x -> -x, V))
        #-----------------------------------------------------------------------
        qswm = size(Vp, 1)
        θ₂ .-= Symmetric(view(Vp, q + 1:qswm, q + 1:qswm))
        mulαtβinc!(βm, view(Vp, 1:q, q + 1:qswm), view(lmm.data.yv, lmm.data.block[i]))
        #-----------------------------------------------------------------------
    end
    mul!(β, inv(θ₂), βm)
    @simd for i = 1:n
        #r    =  lmm.data.yv[i] - lmm.data.xv[i] * β
        #θ₃  += r' * V⁻¹[i] * r
        @inbounds θ₃  += mulθ₃(view(lmm.data.yv, lmm.data.block[i]), view(lmm.data.xv, lmm.data.block[i],:), β, V⁻¹[i])
    end

    logdetθ₂ = logdet(θ₂)

    return   θ₁ + logdetθ₂ + θ₃ + c,  β, θ₂
end

function reml_sweep_β2(lmm::LMM{T2}, θ::Vector{T})::Tuple{T, Vector{T}, Matrix{T}} where T <: Number where T2 <: Number
    n::Int        = length(lmm.data.block)
    N::Int        = length(lmm.data.yv)
    #G::Matrix{T}  = gmat_base(θ, lmm.covstr)
    c::Float64    = (N - lmm.rankx)*log(2π)
    #---------------------------------------------------------------------------
    V⁻¹           = Vector{Matrix{T}}(undef, n)
    # Vector log determinant of V matrix
    θ₁::T         = zero(T)
    θ₂::Matrix{T} = zeros(T, lmm.rankx, lmm.rankx)
    θ₃::T         = zero(T)
    βm::Vector{T} = zeros(T, lmm.rankx)
    β::Vector{T}  = zeros(T, lmm.rankx)

    local q::Int
    local qswm::Int
    local R::Matrix{T}
    local Vp::Matrix{T}
    local logdetθ₂::T

    V   = zeros(T, N, N)
    gmat_base_z!(V, θ, lmm.covstr)
    rmat_basep_z!(V, θ[lmm.covstr.tr[end]], lmm.covstr.rz, lmm.covstr)
    @inbounds for i = 1:n
        q    = length(lmm.data.block[i])
        qswm = q+lmm.rankx
        Vi   = view(V, lmm.data.block[i], lmm.data.block[i])
        θ₁  += logdet(Vi)
        Vp   = zeros(T, q+lmm.rankx, q+lmm.rankx)
        Vpv  = view(Vp, 1:q, 1:q)
        Vpv .= Vi
        Vx   = view(Vp, 1:q, q+1:q+lmm.rankx)
        #Vxt  = view(Vp,  N+1:qswm, 1:N)
        Vx  .= view(lmm.data.xv,  lmm.data.block[i],:)

        sweep!(Vp, 1:q)
        V⁻¹[i] = Symmetric(utriaply!(x -> -x, Vpv))
        #-----------------------------------------------------------------------
        qswm = size(Vp, 1)
        θ₂ .-= Symmetric(view(Vp, q + 1:qswm, q + 1:qswm))
        mulαtβinc!(βm, view(Vp, 1:q, q + 1:qswm), view(lmm.data.yv, lmm.data.block[i]))
        #-----------------------------------------------------------------------
    end
    mul!(β, inv(θ₂), βm)
    @simd for i = 1:n
        #r    =  lmm.data.yv[i] - lmm.data.xv[i] * β
        #θ₃  += r' * V⁻¹[i] * r
        @inbounds θ₃  += mulθ₃(view(lmm.data.yv, lmm.data.block[i]), view(lmm.data.xv, lmm.data.block[i],:), β, V⁻¹[i])
    end

    logdetθ₂ = logdet(θ₂)

    return   θ₁ + logdetθ₂ + θ₃ + c,  β, θ₂
end

function reml_sweep_β3(lmm::LMM{T2}, θ::Vector{T})::Tuple{T, Vector{T}, Matrix{T}} where T <: Number where T2 <: Number
    n::Int        = length(lmm.data.block)
    N::Int        = length(lmm.data.yv)
    G::Matrix{T}  = gmat_base(θ, lmm.covstr)
    c::Float64    = (N - lmm.rankx)*log(2π)
    #---------------------------------------------------------------------------
    V⁻¹           = Vector{Matrix{T}}(undef, n)
    # Vector log determinant of V matrix
    θ₁::T         = zero(T)
    θ₂::Matrix{T} = zeros(T, lmm.rankx, lmm.rankx)
    θ₃::T         = zero(T)
    βm::Vector{T} = zeros(T, lmm.rankx)
    β::Vector{T}  = zeros(T, lmm.rankx)

    local q::Int
    local qswm::Int
    local R::Matrix{T}
    local Vp::Matrix{T}
    local logdetθ₂::T

    @inbounds for i = 1:n
        q   = length(lmm.data.block[i])
        Vp  = zeros(T, q + lmm.rankx, q + lmm.rankx)
        V   = view(Vp, 1:q, 1:q)

        gmat_base_z2!(V, θ, covstr, lmm.data.block[i])
        rmat_basep_z2!(V, θ, view(lmm.covstr.rz, lmm.data.block[i],:), lmm.covstr, lmm.data.block[i])

        #θ₁  += logdet(V)

        θ₁  += logdet(V)

        sweep!(Vp, 1:q)
        V⁻¹[i] = Symmetric(utriaply!(x -> -x, V))
        #-----------------------------------------------------------------------
        qswm = size(Vp, 1)
        θ₂ .-= Symmetric(view(Vp, q + 1:qswm, q + 1:qswm))
        mulαtβinc!(βm, view(Vp, 1:q, q + 1:qswm), view(lmm.data.yv, lmm.data.block[i]))
        #-----------------------------------------------------------------------
    end
    mul!(β, inv(θ₂), βm)
    @simd for i = 1:n
        #r    =  lmm.data.yv[i] - lmm.data.xv[i] * β
        #θ₃  += r' * V⁻¹[i] * r
        @inbounds θ₃  += mulθ₃(view(lmm.data.yv, lmm.data.block[i]), view(lmm.data.xv, lmm.data.block[i],:), β, V⁻¹[i])
    end

    logdetθ₂ = logdet(θ₂)

    return   θ₁ + logdetθ₂ + θ₃ + c,  β, θ₂
end
#=
function reml_sweep_β_ub(lmm::LMM{T2}, θ::Vector{T})::Tuple{T, Vector{T}, Matrix{T}} where T <: Number where T2 <: Number
    n::Int        = length(lmm.data.block)
    N::Int        = length(lmm.data.yv)
    #ZGZ::Matrix{T}= gmat_base_z(θ, lmm.covstr)
    c::Float64    = (N - lmm.rankx)*log(2π)
    #---------------------------------------------------------------------------
    #V⁻¹           = Matrix{T}(undef, n)
    # Vector log determinant of V matrix
    θ₁::T         = zero(T)
    θ₂::Matrix{T} = zeros(T, lmm.rankx, lmm.rankx)
    θ₃::T         = zero(T)
    βm::Vector{T} = zeros(T, lmm.rankx)
    β::Vector{T}  = zeros(T, lmm.rankx)

    qswm = N + lmm.rankx
    Vp   = zeros(T, qswm, qswm)
    Vx   = view(Vp, 1:N, N+1:qswm)
    #Vxt  = view(Vp,  N+1:qswm, 1:N)
    Vx  .= lmm.data.xv
    #Vxt .= lmm.data.xv'

    V   = view(Vp, 1:N, 1:N)
    gmat_base_z!(V, θ, lmm.covstr)
    rmat_basep_z!(V, θ[lmm.covstr.tr[end]], lmm.data.zrv, lmm.covstr)
    θ₁  += logdet(V)

    sweep!(Vp, 1:N)

    V⁻¹ = Symmetric(utriaply!(x -> -x, V))
        #-----------------------------------------------------------------------
    θ₂ .-= Symmetric(view(Vp, N + 1:qswm, N + 1:qswm))
    mulαtβinc!(βm, view(Vp, 1:N, N + 1:qswm), lmm.data.yv)
        #-----------------------------------------------------------------------

    mul!(β, inv(θ₂), βm)
        #r    =  lmm.data.yv[i] - lmm.data.xv[i] * β
        #θ₃  += r' * V⁻¹[i] * r
    @inbounds θ₃  += mulθ₃(lmm.data.yv, lmm.data.xv, β, V⁻¹)


    logdetθ₂ = logdet(θ₂)

    return   θ₁ + logdetθ₂ + θ₃ + c,  β, θ₂
end
=#
################################################################################
function reml_inv_β(lmm::LMM{T2}, θ::Vector{T})::Tuple{T, Vector{T}, Matrix{T}} where T <: Number where T2 <: Number
    n::Int        = length(lmm.data.block)
    N::Int        = length(lmm.data.yv)
    G::Matrix{T}  = gmat_base(θ, lmm.covstr)
    c::Float64    = (N - lmm.rankx)*log(2π)
    #---------------------------------------------------------------------------
    V⁻¹           = Vector{Matrix{T}}(undef, n)
    # Vector log determinant of V matrix
    θ₁::T         = zero(T)
    θ₂::Matrix{T} = zeros(T, lmm.rankx, lmm.rankx)
    θ₃::T         = zero(T)
    βm::Vector{T} = zeros(T, lmm.rankx)
    β::Vector{T}  = zeros(T, lmm.rankx)

    local q::Int
    local qswm::Int
    local R::Matrix{T}
    local Vp::Matrix{T}
    local logdetθ₂::T

    @inbounds for i = 1:n
        q   = length(lmm.data.block[i])
        V  = mulαβαt(view(lmm.data.zv, lmm.data.block[i],:), G)
        rmat_basep!(V, θ[lmm.covstr.tr[end]], view(lmm.data.zrv, lmm.data.block[i],:), lmm.covstr)
        #θ₁  += logdet(V)
        try
            θ₁  += logdet(V)
        catch
            θ₁  += Inf
        end
        V⁻¹[i] = inv(V)
        #-----------------------------------------------------------------------
        #θ2 += Xv[i]'*iVv[i]*Xv[i]
        #βm += Xv[i]'*iVv[i]*yv[i]
        mulθβinc!(θ₂, βm, view(lmm.data.xv, lmm.data.block[i],:), V⁻¹[i], view(lmm.data.yv, lmm.data.block[i]))
        #-----------------------------------------------------------------------
    end
    mul!(β, inv(θ₂), βm)
    @simd for i = 1:n
        #r    =  lmm.data.yv[i] - lmm.data.xv[i] * β
        #θ₃  += r' * V⁻¹[i] * r
        @inbounds θ₃  += mulθ₃(view(lmm.data.yv, lmm.data.block[i]), view(lmm.data.xv, lmm.data.block[i],:), β, V⁻¹[i])
    end
    try
        logdetθ₂ = logdet(θ₂)
    catch
        logdetθ₂ = Inf
    end
    return   θ₁ + logdetθ₂ + θ₃ + c,  β, θ₂
end

################################################################################
