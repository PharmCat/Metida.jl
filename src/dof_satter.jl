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
    if isnothing(lmm.result.h)
        lmm.result.h = reml_hessian(lmm)
    end
    H = lmm.result.h
    n = thetalength(lmm)

    for i = 1:n
        if lmm.covstr.ct[i] == :rho && 1.0 - abs(theta[i]) <= 1E-6
            lmmlog!(lmm, 0, LMMLogMsg(:WARN,
            "Theta parameter $(i): 1 − |ρ̂| ≤ 1e-6 , results can be unstable."))
        end
    end

    if !all(isfinite, H)
        lmmlog!(lmm, 0, LMMLogMsg(:ERROR,
            "Hessian contains non-finite values; Satterthwaite DF not available."))
        return fill(T(NaN), n, n)
    end

    F     = eigen(Symmetric((H .+ H') ./ 2))
    scale = maximum(abs, F.values)
    tol   = scale * sqrt(eps(T)) * n
    keep = findall(x -> x > tol, F.values)

    if length(keep) < n
        lmmlog!(lmm, 0, LMMLogMsg(:WARN,
            "Hessian rank deficient or not positive definite: $(length(keep)) of $n directions retained; Satterthwaite DF computed on a reduced set."))
    end
    if isempty(keep)
        lmmlog!(lmm, 0, LMMLogMsg(:ERROR,
            "Hessian has no usable directions; Satterthwaite DF not available."))
        return fill(T(NaN), n, n)
    end

    Q = view(F.vectors, :, keep)
    return 2 .* (Q * Diagonal(one(T) ./ view(F.values, keep)) * Q')
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
    A     = getinvhes(lmm)
    grad  = gradc(lmm, lmm.result.theta)
    g  = Vector{T}(undef, length(grad))
    for i = 1:length(grad)
        g[i] = dot(l, grad[i], l)
    end
    #d = g' * A * g
    d = dot(g, A, g)
    df = 2*(dot(l, lmm.result.c, l))^2 / d
    if df <= 0
        lmmlog!(lmm, 0, "DF <= 0, indefinite matrix A")
        return T(NaN)
    elseif df < 1.0
        return one(T) 
    elseif df > dof_residual(lmm) 
        return dof_residual(lmm) 
    else 
        return df 
    end
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
        if isnothing(ind) return T(NaN) end
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
    A        = getinvhes(lmm)
    grad     = gradc(lmm, lmm.result.theta)
    dof      = Vector{T}(undef, coefn(lmm))
    fill!(dof, T(NaN))
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
        if !(d > 0)
            lmmlog!(lmm, 0, LMMLogMsg(:WARN, "Zero or non-finite variance of contrast estimate."))
            dof[dofn] = T(NaN); continue
        end
        df = 2*(dot(l, lmm.result.c, l))^2 / d
        dofn = lmm.pivotvec[gi]
        if df <= 0
            lmmlog!(lmm, 0, "DF <= 0, indefinite matrix A")
            dof[dofn] = T(NaN)
        elseif df < 1.0 
            dof[dofn] = 1.0 
        elseif df > dof_residual(lmm) 
            dof[dofn] = dof_residual(lmm) 
        else 
            dof[dofn] = df 
        end
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
    A     = getinvhes(lmm)
    grad  = gradc(lmm, lmm.result.theta)
    g     = Vector{T}(undef, length(grad))

    lcl   = l * lmm.result.c * l'
    lclr  = rank(lcl)
    lclr == 0 && return T(NaN)              # контраст вырожден полностью

    lcls  = Symmetric((lcl .+ lcl') ./ 2)
    lcle  = eigen(lcls)                     
    pl    = lcle.vectors' * l               

    nl    = size(lcls, 1)
    rng   = (nl - lclr + 1):nl

    vm    = Vector{T}(undef, lclr)
    em    = zero(T)
    for (i, k) in enumerate(rng)
        plm = view(pl, k, :)
        for i2 = 1:length(grad)
            g[i2] = dot(plm, grad[i2], plm)
        end
        d     = dot(g, A, g)
        vm[i] = 2 * lcle.values[k]^2 / d
        if vm[i] > 2.0
            em += vm[i] / (vm[i] - 2.0)
        end
    end

    df = 2em / (em - lclr)
    if df <= 0
        lmmlog!(lmm, 0, "DF <= 0, indefinite matrix A")
        return T(NaN)
    elseif df < 1.0
        return one(T) 
    elseif df > dof_residual(lmm)
        return T(dof_residual(lmm))
    else
        return df
    end
end