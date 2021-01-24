#reml.jl

"""
    -2 log Restricted Maximum Likelihood; β calculation inside
"""
function reml_sweep_β_b(lmm, θ::Vector{T}) where T <: Number
    n             = length(lmm.data.block)
    N             = length(lmm.data.yv)
    G             = gmat_base(θ, lmm.covstr)
    c             = (N - lmm.rankx)*log(2π)
    #---------------------------------------------------------------------------
    V⁻¹           = Vector{Matrix{T}}(undef, n)
    # Vector log determinant of V matrix
    θ₁            = zero(T)
    θ₂            = zeros(T, lmm.rankx, lmm.rankx)
    θ₃            = zero(T)
    βm            = zeros(T, lmm.rankx)
    β             = zeros(T, lmm.rankx)
    q             = zero(Int)
    qswm          = zero(Int)
    Vp            = Matrix{T}(undef, 0, 0)
    logdetθ₂      = zero(T)
    @inbounds for i = 1:n
        q   = length(lmm.data.block[i])
        Vp  = mulαβαt3(view(lmm.covstr.z, lmm.data.block[i],:), G, view(lmm.data.xv, lmm.data.block[i],:))
        V   = view(Vp, 1:q, 1:q)
        rmat_base_inc_b!(V, θ[lmm.covstr.tr[end]], view(lmm.covstr.rz, lmm.data.block[i],:), lmm.covstr)
        try
            θ₁  += logdet(cholesky(V))
        catch
            θ₁  = Inf
        end
        sweep!(Vp, 1:q)
        V⁻¹[i] = Symmetric(utriaply!(x -> -x, V))
        #-----------------------------------------------------------------------
        qswm = size(Vp, 1)
        θ₂ -= Symmetric(view(Vp, q + 1:qswm, q + 1:qswm))
        mulαtβinc!(βm, view(Vp, 1:q, q + 1:qswm), view(lmm.data.yv, lmm.data.block[i]))
        #-----------------------------------------------------------------------
    end
    luθ₂ = lu(θ₂)
    ldiv!(β, luθ₂, βm)
    @simd for i = 1:n
        @inbounds θ₃  += mulθ₃(view(lmm.data.yv, lmm.data.block[i]), view(lmm.data.xv, lmm.data.block[i],:), β, V⁻¹[i])
    end
    try
        logdetθ₂ = logdet(θ₂)
    catch
        logdetθ₂ = Inf
    end
    return   θ₁ + logdetθ₂ + θ₃ + c, β, θ₂, θ₃
end

function reml_sweep_β(lmm, θ::Vector{T}) where T <: Number
    n             = length(lmm.data.block)
    N             = length(lmm.data.yv)
    c             = (N - lmm.rankx)*log(2π)
    #---------------------------------------------------------------------------
    V⁻¹           = Vector{Matrix{T}}(undef, n)
    # Vector log determinant of V matrix
    θ₁            = zero(T)
    θ₂            = zeros(T, lmm.rankx, lmm.rankx)
    θ₃            = zero(T)
    βm            = zeros(T, lmm.rankx)
    β             = Vector{T}(undef, lmm.rankx)
    #---------------------------------------------------------------------------
    q             = zero(Int)
    qswm          = zero(Int)
    Vp            = Matrix{T}(undef, 0, 0)
    logdetθ₂      = zero(T)

    @inbounds for i = 1:n
        q    = length(lmm.data.block[i])
        qswm = q + lmm.rankx
        Vp  = zeros(T, q + lmm.rankx, q + lmm.rankx)
        V   = view(Vp, 1:q, 1:q)
        Vx   = view(Vp, 1:q, q+1:q+lmm.rankx)
        Vx  .= view(lmm.data.xv,  lmm.data.block[i],:)
        zgz_base_inc!(V, θ, lmm.covstr, lmm.data.block[i], lmm.covstr.sblock[i])
        rmat_base_inc!(V, θ[lmm.covstr.tr[end]], lmm.covstr, lmm.data.block[i], lmm.covstr.sblock[i])
        #-----------------------------------------------------------------------
        θ₁  += logdet(cholesky(V))
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
        @inbounds θ₃  += mulθ₃(view(lmm.data.yv, lmm.data.block[i]), view(lmm.data.xv, lmm.data.block[i],:), β, V⁻¹[i])
    end
    logdetθ₂ = logdet(θ₂)
    return   θ₁ + logdetθ₂ + θ₃ + c, β, θ₂, θ₃
end
################################################################################
################################################################################
