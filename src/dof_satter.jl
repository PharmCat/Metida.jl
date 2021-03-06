#dof_satter.jl

function gradc(lmm::LMM{T}, theta) where T
    if !lmm.result.fit error("Model not fitted!") end
    vloptf(x) = sweep_β_cov(lmm, x, lmm.result.beta)
    chunk  = ForwardDiff.Chunk{1}()
    jcfg   = ForwardDiff.JacobianConfig(vloptf, theta, chunk)
    jic    = ForwardDiff.jacobian(vloptf, theta, jcfg, Val{false}())
    grad   = Vector{Matrix{T}}(undef, thetalength(lmm))
    for i in 1:thetalength(lmm)
        gic     = reshape(view(jic, :, i), rankx(lmm), rankx(lmm)) #<Opt
        grad[i] = - lmm.result.c * gic * lmm.result.c
    end
    grad
end

function getinvhes(lmm)
    local A
    if isnothing(lmm.result.h)
        lmm.result.h = hessian(lmm)
        H = copy(lmm.result.h)
    else
        H = copy(lmm.result.h)
    end
    theta = copy(lmm.result.theta)
    qrd   = qr(H, Val(true))
    vals  = falses(thetalength(lmm))
    for i = 1:thetalength(lmm)
        if lmm.covstr.ct[qrd.jpvt[i]] == :var
            if abs(qrd.R[i, i]) > 1E-8
                vals[qrd.jpvt[i]] = true
            else
                theta[qrd.jpvt[i]] = 0.0
                H[:,qrd.jpvt[i]]  .= 0.0
                H[qrd.jpvt[i],:]  .= 0.0
            end
        elseif lmm.covstr.ct[qrd.jpvt[i]] == :rho
            if 1.0 - abs(lmm.result.theta[qrd.jpvt[i]])  > 1E-6
                vals[qrd.jpvt[i]] = true
            else
                if lmm.result.theta[qrd.jpvt[i]] > 0 lmm.result.theta[qrd.jpvt[i]] = 1.0 else lmm.result.theta[qrd.jpvt[i]] = -1.0 end
                H[:,qrd.jpvt[i]] .= 0.0
                H[qrd.jpvt[i],:] .= 0.0
            end
        end
    end
    try
        vh  = view(H, vals, vals)
        vh .= inv(Matrix(vh))
        A = 2.0 * H
    catch
        A = 2.0 * pinv(H)
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
function dof_satter(lmm::LMM{T}, l::Vector) where T
    A, theta = getinvhes(lmm)
    grad  = gradc(lmm, theta)
    g  = Vector{T}(undef, length(grad))
    for i = 1:length(grad)
        g[i] = (l' * grad[i] * l)[1]
    end
    d = g' * A * g
    df = 2*(l' * lmm.result.c * l)^2 / d
    if df < 1.0 return 1.0 elseif df > dof_residual(lmm) return dof_residual(lmm) else return df end
end
"""
    dof_satter(lmm::LMM{T}, n::Int) where T

Return Satterthwaite approximation for the denominator degrees of freedom, where `n` - coefficient number.
"""
function dof_satter(lmm::LMM{T}, n::Int) where T
    l = zeros(Int, length(lmm.result.beta))
    l[n] = 1
    dof_satter(lmm, l)
end
"""
    dof_satter(lmm::LMM{T}) where T

Return Satterthwaite approximation for the denominator degrees of freedom for all coefficients.

"""
function dof_satter(lmm::LMM{T}) where T
    isfitted(lmm) || error("Model not fitted")
    lb       = length(lmm.result.beta)
    A, theta = getinvhes(lmm)
    grad     = gradc(lmm, theta)
    dof      = Vector{Float64}(undef, lb)
    for gi = 1:lb
        l     = zeros(Int, lb)
        l[gi] = 1
        g     = Vector{T}(undef, length(grad))
        for i = 1:length(grad)
            g[i] = (l' * grad[i] * l)[1]
        end
        d = g' * A * g
        df = 2*(l' * lmm.result.c * l)^2 / d
        if df < 1.0 dof[gi] = 1.0 elseif df > dof_residual(lmm) dof[gi] = dof_residual(lmm) else dof[gi] = df end
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
function dof_satter(lmm::LMM{T}, l::Matrix) where T
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
            g[i2] = (plm' * grad[i2] * plm)[1]
        end
        d = g' * A * g
        vm[i] = 2*lcle.values[i]^2 / d
        if vm[i] > 2.0 em += vm[i] / (vm[i] - 2.0) end
    end
    df = 2em/(em - lclr)
    if df < 1.0 return 1.0 elseif df > dof_residual(lmm) return dof_residual(lmm) else return df end
end
