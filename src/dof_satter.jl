#dof_satter.jl

function gradc(lmm::LMM{T}, theta) where T
    if !lmm.result.fit error("Model not fitted!") end
    if !isnothing(lmm.result.grc) return lmm.result.grc end
    vloptf(x) = sweep_β_cov(lmm, lmm.dv, x, lmm.result.beta)
    chunk  = ForwardDiff.Chunk{min(10, length(theta))}()
    jcfg   = ForwardDiff.JacobianConfig(vloptf, theta, chunk)
    jic    = ForwardDiff.jacobian(vloptf, theta, jcfg)
    grad   = Vector{Matrix{T}}(undef, thetalength(lmm))
    for i in 1:thetalength(lmm)
        gic     = reshape(view(jic, :, i), rankx(lmm), rankx(lmm)) #<Opt
        grad[i] = - lmm.result.c * gic * lmm.result.c
    end
    lmm.result.grc = grad
    grad
end

function getinvhes(lmm::LMM{T}) where T
    local A
    if isnothing(lmm.result.h)
        lmm.result.h = hessian(lmm)
        H = copy(lmm.result.h)
    else
        H = copy(lmm.result.h)
    end
    theta = copy(lmm.result.theta)
    qrd   = qr(H)
    vals  = falses(thetalength(lmm))
    for i = 1:thetalength(lmm)
        if lmm.covstr.ct[i] == :rho
            if 1.0 - abs(lmm.result.theta[i])  > 1E-6
                vals[i] = true
            else
                if lmm.result.theta[i] > 0 lmm.result.theta[i] = 1.0 else lmm.result.theta[i] = -1.0 end
                H[:,i] .= zero(T)
                H[i,:] .= zero(T)
            end
        else
            if abs(qrd.R[i, i]) > 1E-8
                vals[i] = true
            else
                theta[i] = zero(T)
                H[:,i]  .= zero(T)
                H[i,:]  .= zero(T)
            end
        end
    end
    try
        vh  = view(H, vals, vals)
        vh .= inv(Matrix(vh))
        A = H * 2
    catch
        A = pinv(H) * 2
    end
    A, theta
end
"""
    dof_satter(lmm::LMM{T}, l) where T

Return Satterthwaite approximation for the denominator degrees of freedom, where `l` is a contrast vector (estimable linear combination
of fixed effect coefficients vector (`β`).

```math
df = \\frac{2(LCL')^{2}}{g'Ag}
```

Where: ``A = 2H^{-1}``, ``g = \\triangledown_{\\theta}(LC^{-1}_{\\theta}L')``

"""
function dof_satter(lmm::LMM{T}, l::AbstractVector) where T
    isfitted(lmm) || error("Model not fitted")
    dof_satter_(lmm, ifelse(lmm.rankx == coefn(lmm), l, view(l, lmm.pivotvec)))
end

function dof_satter_(lmm::LMM{T}, l::AbstractVector) where T
    A, theta = getinvhes(lmm)
    grad  = gradc(lmm, theta)
    g  = Vector{T}(undef, length(grad))
    for i = 1:length(grad)
        g[i] = dot(l, grad[i], l)
    end
    #d = g' * A * g
    d = dot(g, A, g)
    df = 2*(dot(l, lmm.result.c, l))^2 / d
    if df < 1.0 return 1.0 elseif df > dof_residual(lmm) return dof_residual(lmm) else return df end
end
"""
    dof_satter(lmm::LMM{T}, i::Int) where T

Return Satterthwaite approximation for the denominator degrees of freedom, where `n` - coefficient number.
"""
function dof_satter(lmm::LMM{T}, i::Int) where T
    isfitted(lmm) || error("Model not fitted")
    if coefn(lmm) == lmm.rankx
        ind = i
    else
        ind = findfirst(x-> x == i, lmm.pivotvec)
        if isnothing(ind) return NaN end
    end
    l = zeros(T, lmm.rankx)
    l[ind] = one(T)
    return dof_satter(lmm, l)
end
"""
    dof_satter(lmm::LMM{T}) where T

Return Satterthwaite approximation for the denominator degrees of freedom for all coefficients.

"""
function dof_satter(lmm::LMM{T}) where T
    isfitted(lmm) || error("Model not fitted")
    lb       = lmm.rankx
    A, theta = getinvhes(lmm)
    grad     = gradc(lmm, theta)
    dof      = Vector{T}(undef, coefn(lmm))
    fill!(dof, NaN)
    l        = Vector{T}(undef, lb)
    for gi = 1:lb
        fill!(l, zero(T))
        l[gi] = one(T)
        g     = Vector{T}(undef, length(grad))
        for i = 1:length(grad)
            g[i] = dot(l, grad[i], l)
        end
        #d = g' * A * g
        d = dot(g, A, g)
        df = 2*(dot(l, lmm.result.c, l))^2 / d
        dofn = lmm.pivotvec[gi]
        if df < 1.0 dof[dofn] = 1.0 elseif df > dof_residual(lmm) dof[dofn] = dof_residual(lmm) else dof[dofn] = df end
    end
    dof
end

"""
    dof_satter(lmm::LMM{T}, l::Matrix) where T

Return Satterthwaite approximation for the denominator degrees of freedom for conrast matrix `l`.

For `size(l, 1)` > 1:

```math
df = \\frac{2E}{E - rank(LCL')}
```

where:

* let ``LCL' = QΛQ^{-1}``, where ``QΛQ^{-1}`` - spectral decomposition of ``LCL'``
* ``Lq_i`` is the i-th row of ``Q^{-1}L``
* ``A = 2H^{-1}``, ``g = \\triangledown_{\\theta}(Lq_i C^{-1}_{\\theta} Lq_i')``
* ``v_i = \\frac{2*Λ_{i,i}^2}{g' * A * g}``
* ``E = \\sum_{i=1}^n {\\frac{v_i}(v_i - 2)}`` for ``v_i > 2``
"""
function dof_satter(lmm::LMM{T}, l::AbstractMatrix) where T
    isfitted(lmm) || error("Model not fitted")
    if coefn(lmm) != size(l, 2) error("size(l, 2) not equal rank X!") end
    dof_satter_(lmm, ifelse(lmm.rankx == coefn(lmm), l, view(l, :, lmm.pivotvec)))
end

function dof_satter_(lmm::LMM{T}, l::AbstractMatrix) where T
    A, theta = getinvhes(lmm)
    grad  = gradc(lmm, theta)
    g     = Vector{T}(undef, length(grad))

    lcl   = l*lmm.result.c*l'
    lclr  = rank(lcl)
    #if lclr != size(l, 1) error() end
    lcle  = eigen(lcl)
    pl    = lcle.vectors'*l
    vm    = Vector{T}(undef, lclr)
    em    = 0
    for i = 1:lclr
        plm = pl[i,:]
        for i2 = 1:length(grad)
            g[i2] = dot(plm, grad[i2], plm)
        end
        #d = g' * A * g
        d = dot(g, A, g)
        vm[i] = 2*lcle.values[i]^2 / d
        if vm[i] > 2.0 em += vm[i] / (vm[i] - 2.0) end
    end
    df = 2em/(em - lclr)
    if df < 1.0 return 1.0 elseif df > dof_residual(lmm) return dof_residual(lmm) else return df end
end
