#reml.jl

"""
    2 log Restricted Maximum Likelihood
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

function reml_sweep(yv, Zv, p, Xv, θvec, β)
    n = length(yv)
    N = sum(length.(yv))
    G = gmat(θvec[3:5])
    c  = (N-p)*log(2π)
    θ1 = 0.0
    θ2 = 0.0
    θ3 = 0.0
    θ2m  = zeros(eltype(θvec), p, p)

    for i = 1:n
        q = length(yv[i])
        r = mulr(yv[i], Xv[i], β)
        R   = rmat([θvec[1], θvec[2]], Zv[i])
        Vp  = mulαβαtc2(Zv[i], G, R, r)
        V = view(Vp, 1:q, 1:q)
        θ1  += logdet(V)
        sweep!(Vp, 1:q)
        iV  = Symmetric(-Vp[1:q, 1:q])
        mulαtβαinc!(θ2m, Xv[i], iV)
        θ3  += -Vp[end, end]
    end
    θ2       = logdet(θ2m)
    return   -(θ1 + θ2 + θ3 + c)
end

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
