#reml.jl

"""
    -2 log Restricted Maximum Likelihood
"""
function reml(yv, Zv, p, Xv, θvec, β)
    n = length(yv)
    N = sum(length.(yv))
    G = gmat(θvec[3:5])
    c  = (N-p)*log(2π)
    θ1 = 0
    θ2 = 0
    θ3 = 0
    iV   = nothing
    θ2m  = zeros(eltype(θvec), p, p)
    for i = 1:n
        R   = rmat([θvec[1], θvec[2]], Zv[i])
        V   = vmat(G, R, Zv[i])
        iV  = inv(V)
        θ1  += logdet(V)
        mulαtβαinc!(θ2m, Xv[i], iV)
        θ3  += mulθ₃(yv[i], Xv[i], β, iV)
    end
    θ2       = logdet(θ2m)
    return   -(θ1 + θ2 + θ3 + c)
end

function reml_sweep(lmm, β, θ::Vector{T})::T where T <: Number
    n = length(lmm.data.yv)
    N = sum(length.(lmm.data.yv))
    G = gmat_blockdiag(θ, lmm.covstr)
    c  = (N - lmm.rankx)*log(2π)
    θ₁ = zero(eltype(θ))
    θ₂ = zero(eltype(θ))
    θ₃ = zero(eltype(θ))
    θ2m  = zeros(eltype(θ), lmm.rankx, lmm.rankx)
    @inbounds for i = 1:n
        q   = length(lmm.data.yv[i])
        r   = mulr(lmm.data.yv[i], lmm.data.xv[i], β)
        R   = rmat(θ[lmm.covstr.tr[end]], lmm.data.zrv[i], q, lmm.covstr.repeated)
        Vp  = mulαβαtc2(lmm.data.zv[i], G, R, r)
        V   = view(Vp, 1:q, 1:q)
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

function reml_sweep_β(lmm::LMM{T2}, θ::Vector{T})::Tuple{T, Vector{T}, Matrix{T}} where T <: Number where T2 <: Number
    n::Int        = length(lmm.data.yv)
    N::Int        = sum(length.(lmm.data.yv))
    G             = gmat_blockdiag(θ, lmm.covstr)
    c::Float64    = (N - lmm.rankx)*log(2π)
    #---------------------------------------------------------------------------
    V⁻¹           = Vector{Matrix{T}}(undef, n)
    # Vector log determinant of V matrix
    θ₁::T         = zero(T)
    θ₂::Matrix{T} = zeros(promote_type(T2, T), lmm.rankx, lmm.rankx)
    θ₃::T         = zero(T)
    βm::Vector{T} = zeros(promote_type(T2, T), lmm.rankx)
    β::Vector{T}  = zeros(promote_type(T2, T), lmm.rankx)

    local q::Int
    local R::Matrix{T}
    local Vp::Matrix{T}

    @inbounds for i = 1:n
        q   = length(lmm.data.yv[i])
        #R   = rmatbase(lmm, q, i, θ[lmm.covstr.tr[end]])
        R   = rmat(θ[lmm.covstr.tr[end]], lmm.data.zrv[i], q, Val{lmm.covstr.ves[end]}())
        Vp  = mulαβαtc3(lmm.data.zv[i], G, R, lmm.data.xv[i])
        V   = view(Vp, 1:q, 1:q)
        θ₁  += logdet(V)
        sweep!(Vp, 1:q)
        V⁻¹[i] = Symmetric(utriaply!(x -> -x, Vp[1:q, 1:q]))
        #-----------------------------------------------------------------------
        θ₂ .-= Symmetric(Vp[q + 1:end, q + 1:end])
        mulαtβinc!(βm, Vp[1:q, q + 1:end], lmm.data.yv[i])
        #mulθβinc!(θ₂, βm, data.Xv[i], V⁻¹[i], data.yv[i], first(data.mem.svec))
        #-----------------------------------------------------------------------
    end
    mul!(β, inv(θ₂), βm)
    @simd for i = 1:n
        #r    =  lmm.data.yv[i] - lmm.data.xv[i] * β
        #θ₃  += r' * V⁻¹[i] * r
        @inbounds θ₃  += mulθ₃(lmm.data.yv[i], lmm.data.xv[i], β, V⁻¹[i])
    end
    return   θ₁ + logdet(θ₂) + θ₃ + c,  β, θ₂
end

################################################################################


################################################################################


"""
    2 log Restricted Maximum Likelihood gradient vector
"""

function reml_grad(yv, Zv, p, Xv, θvec, β)
    n     = length(yv)
    G     = gmat(θvec[3:5])
    θ1    = zeros(length(θvec))
    θ2    = zeros(length(θvec))
    θ3    = zeros(length(θvec))
    iV    = Vector{AbstractMatrix}(undef, n)
    θ2m   = zeros(p,p)
    H     = zeros(p, p)
    for i = 1:n
        iV[i] = inv(vmat(G, rmat(θvec[1:2], Zv[i]), Zv[i]))
        mulαtβαinc!(H, Xv[i], iV[i])
        #H .+= Xv[i]'*inv(vmat(G, rmat(θvec[1:2], Zv[i]), Zv[i]))*Xv[i]
    end
    iH = inv(H)
    #fx = x -> vmat(gmat(x[3:5]), rmat(x[1:2], Zv[1]), Zv[1])
    #cfg   = ForwardDiff.JacobianConfig(fx, θvec)
    for i = 1:n
        #V   = vmat(G, rmat(θvec[1:2], Zv[i]), Zv[i])
        #iV  = inv(V)
        r   = yv[i] .- Xv[i]*β
        jV  = covmat_grad(vmat, Zv[i], θvec)
        Aj  = zeros(length(yv[i]), length(yv[i]))
        for j = 1:length(θvec)

            mulαβαc!(Aj, iV[i], view(jV, :, :, j))
            #Aj      = iV[i] * view(jV, :, :, j) * iV[i]

            θ1[j]  += trmulαβ(iV[i], view(jV, :, :, j))
            #θ1[j]  += tr(iV[i] * view(jV, :, :, j))

            θ2[j]  -= tr(iH * Xv[i]' *Aj * Xv[i])

            θ3[j]  -= r' * Aj * r
        end
    end
    return - (θ1 .+ θ2 .+ θ3)
end

"""
    2 log Restricted Maximum Likelihood hessian matrix
"""
function reml_hessian(yv, Zv, p, Xv, θvec, β)
    n     = length(yv)
    G     = gmat(θvec[3:5])
    θ1    = zeros(length(θvec))
    θ2    = zeros(length(θvec))
    θ3    = zeros(length(θvec))
    iV    = nothing
    θ2m   = zeros(p,p)
    H     = zeros(p, p)
    for i = 1:n
        H += Xv[i]'*inv(vmat(rmat(θvec[1:2], Zv[i]), G, Zv[i]))*Xv[i]
    end
    iH = inv(H)
    for i = 1:n
        vmatdvec = x -> vmat(rmat(x[1:2], Zv[i]), gmat(x[3:end]), Zv[i])[:]
        V   = vmat(rmat(θvec[1:2], Zv[i]), G, Zv[i])
        iV  = inv(V)
        r   = yv[i] .- Xv[i]*β
        ∇V  = covmat_grad(covmat, θ, cfg)
        ∇²V = covmat_hessian(covmat, θ)
        #Aij        = iV*∇V[j]*iV
        #Aijk       = -iV * (∇V[k] * iV * ∇V[j] - ∇²V[k,j] + ∇V[j] * iV * ∇V[k]) * iV

        #θ1[j]  += tr(iV * ∇V[j])
        #θ2[j]  -= tr(iH * Xv[i]' * Aij * Xv[i])
        #θ3[j]  -= r' * Aij * r

        #θ1[j,k]  += tr( - Aik' * ∇V[j] + iV * ∇²V[j,k])
        #θ2[j,k]  = - tr( iH * sum(X' * Aik * X) * iH * sum(X' * Aij * X)) - tr(iH * sum(X' * Aijk * X))
        #θ3[j,k]  -= r' * Aijk * r

        #A[j,k] =

        for j = 1:length(θvec)
            for k = 1:length(θvec)

            #θ1[j,k]  +=

            #θ2[j,k]  -=

            #θ3[j,k]  -=
            end
        end

    end
    return - (θ1 .+ θ2 .+ θ3)
end

################################################################################

function reml_grad2(yv, Zv, p, Xv, θvec, β)
    n     = length(yv)
    for i = 1:n
        jV  = covmat_grad(vmat, Zv[i], θvec)
    end
end
function reml_grad3(yv, Zv, p, Xv, θvec, β)
    n     = length(yv)
    covmat_grad2(vmatvec, Zv, θvec)
end
