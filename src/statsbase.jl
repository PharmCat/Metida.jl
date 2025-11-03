"""
    StatsBase.islinear(model::LMM)

"""
StatsBase.islinear(model::LMM) = true


"""
    StatsBase.confint(lmm::LMM{T}; level::Real=0.95, ddf::Symbol = :satter) where T

Confidece interval for coefficients.

ddf = :satter/:residual

```math
CI_{U/L} = β ± SE * t_{ddf, 1-α/2}
```

See also: [`dof_satter`](@ref), [`dof_residual`](@ref)
"""
function StatsBase.confint(lmm::LMM{T}; level::Real=0.95, ddf::Symbol = :satter) where T
    isfitted(lmm) || error("Model not fitted")
    alpha = 1.0 - level
    if ddf == :satter
        ddfv = dof_satter(lmm)
    elseif ddf == :contain
        ddfv = dof_contain(lmm)
    elseif ddf == :residual
        ddfv = fill!(Vector{Float64}(undef, coefn(lmm)), dof_residual(lmm))
    end
    cis = fill((NaN, NaN), coefn(lmm))
    for i = 1:lmm.rankx
        #ERROR: ArgumentError: TDist: the condition ν > zero(ν) is not satisfied
        d = lmm.result.se[i] * quantile(TDist(ddfv[lmm.pivotvec[i]]), 1.0 - alpha / 2)
        cis[lmm.pivotvec[i]] = (lmm.result.beta[i] - d, lmm.result.beta[i] + d)
    end
    cis
end

"""
    StatsBase.confint(lmm::LMM{T}, i::Int; level::Real=0.95, ddf::Symbol = :satter) where T

Confidece interval for coefficient `i`.
"""
function StatsBase.confint(lmm::LMM{T}, i::Int; level::Real=0.95, ddf::Symbol = :satter) where T
    isfitted(lmm) || error("Model not fitted")
    if i < 1 || i > coefn(lmm)
        error("Wrong coef number")
    end
    if coefn(lmm) == lmm.rankx
        ind = i
    else
        ind = findfirst(x-> x == i, lmm.pivotvec)
        if isnothing(ind) return NaN end
    end

    alpha = 1.0 - level
    if ddf == :satter
        ddfv = dof_satter(lmm, i)
    elseif ddf == :contain
        ddfv = dof_contain(lmm, i)
    elseif ddf == :residual
        ddfv = dof_residual(lmm)
    end

    #ERROR: ArgumentError: TDist: the condition ν > zero(ν) is not satisfied
    d = lmm.result.se[ind] * quantile(TDist(ddfv), 1.0 - alpha / 2)
    (lmm.result.beta[ind] - d, lmm.result.beta[ind] + d)

end


#=
REML: n = total number of observation - number fixed effect parameters; d = number of covariance parameters
ML:, n = total number of observation; d = number of fixed effect parameters + number of covariance parameters.
=#
"""
    StatsBase.coef(lmm::LMM) = copy(lmm.result.beta)

Model coefficients (β).
"""
function StatsBase.coef(lmm::LMM{T}) where T
    cn = coefn(lmm)
    if cn == lmm.rankx
        return copy(coef_(lmm))
    else
        v = zeros(T, cn)
        v[lmm.pivotvec] .= coef_(lmm)
        return v
    end
end

function coef_(lmm::LMM)
    return lmm.result.beta
end
"""
    StatsBase.coefnames(lmm::LMM) = StatsBase.coefnames(lmm.mf)

Coefficients names.
"""
StatsBase.coefnames(lmm::LMM) = StatsBase.coefnames(lmm.f)[2]

"""
    StatsBase.nobs(lmm::LMM)

Number of observations.
"""
function StatsBase.nobs(lmm::Union{LMM, MILMM})
    return length(lmm.data.yv)
end

"""
    StatsBase.dof_residual(lmm::LMM)

DOF residuals: N - rank(X), where N - total number of observations.
"""
function StatsBase.dof_residual(lmm::LMM)
    return nobs(lmm) - lmm.rankx
end

"""
    StatsBase.dof(lmm::LMM)

DOF.
"""
function StatsBase.dof(lmm::LMM)
    return lmm.nfixed + lmm.covstr.tl
end

"""
    StatsBase.loglikelihood(lmm::LMM)

Return loglikelihood value.
"""
function StatsBase.loglikelihood(lmm::LMM)
    return -lmm.result.reml/2
end

"""
    StatsBase.aic(lmm::LMM)

Akaike Information Criterion.
"""
function StatsBase.aic(lmm::LMM)
    l = loglikelihood(lmm)
    d = lmm.covstr.tl
    return -2l + 2d
end

"""
    StatsBase.bic(lmm::LMM)

Bayesian information criterion.
"""
function StatsBase.bic(lmm::LMM)
    l = loglikelihood(lmm)
    d = lmm.covstr.tl
    n = nobs(lmm) - lmm.nfixed
    return -2l + d * log(n)
end

"""
    StatsBase.aicc(lmm::LMM)

Corrected Akaike Information Criterion.
"""
function StatsBase.aicc(lmm::LMM)
    l = loglikelihood(lmm)
    d = lmm.covstr.tl
    n = nobs(lmm) - lmm.nfixed
    return -2l + (2d * n) / (n - d - 1.0)
end

"""
    caic(lmm::LMM)

Conditional Akaike Information Criterion.
"""
function caic(lmm::LMM)
    l = loglikelihood(lmm)
    d = lmm.covstr.tl
    n = nobs(lmm) - lmm.nfixed
    return -2l + d * (log(n) + 1.0)
end

"""
    StatsBase.isfitted(lmm::LMM)
"""
function StatsBase.isfitted(lmm::LMM)
    return lmm.result.fit
end
"""
    StatsBase.vcov(lmm::LMM)

Variance-covariance matrix of β.
"""
function StatsBase.vcov(lmm::LMM{T}) where T 
    cn = coefn(lmm) 
    if cn == lmm.rankx
        return copy(lmm.result.c)
    else
        m = Matrix{T}(undef, cn, cn)
        fill!(m, NaN)
        m[lmm.pivotvec, lmm.pivotvec] .= lmm.result.c
        return m
    end
end
"""
    StatsBase.stderror(lmm::LMM)

Standard error
"""
function StatsBase.stderror(lmm::LMM{T}) where T
    cn = coefn(lmm) 
    if cn == lmm.rankx
        return copy(stderror_(lmm))
    else
        v = Vector{T}(undef, cn)
        fill!(v, NaN)
        v[lmm.pivotvec] .= stderror_(lmm)
        return v
    end
end    

function stderror_(lmm::LMM)
    return lmm.result.se
end

function stderror!(v, lmm::LMM) 
    cn = coefn(lmm) 
    if cn == lmm.rankx
        copyto!(v, lmm.result.se)
        return v 
    else
        fill!(v, NaN)
        v[lmm.pivotvec] .= stderror_(lmm)
        return v
    end
end

"""
    StatsBase.modelmatrix(lmm::LMM)

Fixed effects matrix.
"""
StatsBase.modelmatrix(lmm::LMM) = lmm.data.xv
"""
    StatsBase.response(lmm::LMM)

Response vector.
"""
StatsBase.response(lmm::LMM) = lmm.data.yv

"""
    crossmodelmatrix(lmm::LMM)

Return X'X.
"""
StatsBase.crossmodelmatrix(lmm::LMM) = (x = modelmatrix(lmm); Symmetric(x' * x))

"""
    coeftable(lmm::LMM)

Return coefficients table.
"""
function StatsBase.coeftable(lmm::LMM)
    co = coef(lmm)
    se = stderror!(similar(co), lmm)
    z  = co ./ se
    pvalue = ccdf.(Chisq(1), abs2.(z))
    names = coefnames(lmm)
    if !isa(names, AbstractVector) names = [names] end
    return CoefTable(
        hcat(co, se, z, pvalue),
        ["Coef.", "Std. Error", "z", "Pr(>|z|)"],
        names,
        4,
        3,
    )
end

"""
    responsename(lmm::LMM)

Responce varible name.
"""
function StatsBase.responsename(lmm::LMM)
    return StatsBase.coefnames(lmm.f)[1]
end


# This can be supported
#=
StatsBase.weights(model::LMM) = error("weights is not defined for $(typeof(model)).")
StatsBase.residuals(model::LMM) = error("residuals is not defined for $(typeof(model)).")
StatsBase.predict(model::LMM) = error("predict is not defined for $(typeof(model)).")
StatsBase.predict!(model::LMM) = error("predict! is not defined for $(typeof(model)).")
StatsBase.leverage(model::LMM) = error("leverage is not defined for $(typeof(model)).")
StatsBase.deviance(model::LMM) = error("deviance is not defined for $(typeof(model)).")
=#
