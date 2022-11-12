"""
    islinear(model::LMM) = true
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
    cis = Vector{Tuple{T, T}}(undef, coefn(lmm))
    for i = 1:coefn(lmm)
        #ERROR: ArgumentError: TDist: the condition ν > zero(ν) is not satisfied
        d = lmm.result.se[i] * quantile(TDist(ddfv[i]), 1.0 - alpha / 2)
        cis[i] = (lmm.result.beta[i] - d, lmm.result.beta[i] + d)
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
    alpha = 1.0 - level
    if ddf == :satter
        ddfv = dof_satter(lmm, i)
    elseif ddf == :contain
        ddfv = dof_contain(lmm, i)
    elseif ddf == :residual
        ddfv = dof_residual(lmm)
    end

    #ERROR: ArgumentError: TDist: the condition ν > zero(ν) is not satisfied
    d = lmm.result.se[i] * quantile(TDist(ddfv), 1.0 - alpha / 2)
    (lmm.result.beta[i] - d, lmm.result.beta[i] + d)

end


#=
REML: n = total number of observation - number fixed effect parameters; d = number of covariance parameters
ML:, n = total number of observation; d = number of fixed effect parameters + number of covariance parameters.
=#
"""
    StatsBase.coef(lmm::LMM) = copy(lmm.result.beta)

Model coefficients (β).
"""
StatsBase.coef(lmm::LMM) = copy(coef_(lmm))

function coef_(lmm::LMM)
    lmm.result.beta
end
"""
    StatsBase.coefnames(lmm::LMM) = StatsBase.coefnames(lmm.mf)

Coefficients names.
"""
StatsBase.coefnames(lmm::LMM) = StatsBase.coefnames(lmm.mf)

"""
    StatsBase.nobs(lmm::MetiaModel)

Number of observations.
"""
function StatsBase.nobs(lmm::MetidaModel)
    return length(lmm.data.yv)
end

"""
    StatsBase.dof_residual(lmm::LMM)

DOF residuals: N - rank(X), where N - total number of observations.
"""
function StatsBase.dof_residual(lmm::LMM)
    nobs(lmm) - lmm.rankx
end

"""
    StatsBase.dof(lmm::LMM)

DOF.
"""
function StatsBase.dof(lmm::LMM)
    lmm.nfixed + lmm.covstr.tl
end

"""
    StatsBase.loglikelihood(lmm::LMM)

Return loglikelihood value.
"""
function StatsBase.loglikelihood(lmm::LMM)
    -lmm.result.reml/2
end

"""
    StatsBase.aic(lmm::LMM)

Akaike Information Criterion.
"""
function StatsBase.aic(lmm::LMM)
    l = loglikelihood(lmm)
    d = lmm.covstr.tl
    -2l + 2d
end

"""
    StatsBase.bic(lmm::LMM)

Bayesian information criterion.
"""
function StatsBase.bic(lmm::LMM)
    l = loglikelihood(lmm)
    d = lmm.covstr.tl
    n = nobs(lmm) - lmm.nfixed
    -2l + d * log(n)
end

"""
    StatsBase.aicc(lmm::LMM)

Corrected Akaike Information Criterion.
"""
function StatsBase.aicc(lmm::LMM)
    l = loglikelihood(lmm)
    d = lmm.covstr.tl
    n = nobs(lmm) - lmm.nfixed
    -2l + (2d * n) / (n - d - 1.0)
end

"""
    caic(lmm::LMM)

Conditional Akaike Information Criterion.
"""
function caic(lmm::LMM)
    l = loglikelihood(lmm)
    d = lmm.covstr.tl
    n = nobs(lmm) - lmm.nfixed
    -2l + d * (log(n) + 1.0)
end

"""
    StatsBase.isfitted(lmm::LMM)
"""
function StatsBase.isfitted(lmm::LMM)
    lmm.result.fit
end
"""
    StatsBase.vcov(lmm::LMM)

Variance-covariance matrix of β.
"""
StatsBase.vcov(lmm::LMM) = copy(lmm.result.c)
"""
    StatsBase.stderror(lmm::LMM)

Standard error
"""
StatsBase.stderror(lmm::LMM) = copy(stderror_(lmm))
#StatsBase.stderror(lmm::LMM) = sqrt.(diag(vcov(lmm)))

function stderror_(lmm::LMM)
    lmm.result.se
end

stderror!(v, lmm::LMM) = copyto!(v, lmm.result.se)

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

Return responce name as string.
"""
function StatsBase.responsename(lmm::LMM)
    string(lmm.model.lhs.sym)
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

# Not supported
#=
StatsBase.nulldeviance(model::LMM) =
    error("nulldeviance is not defined for $(typeof(model)).")

StatsBase.nullloglikelihood(model::LMM) =
        error("nullloglikelihood is not defined for $(typeof(model)).")

StatsBase.score(model::LMM) = error("score is not defined for $(typeof(model)).")

StatsBase.mss(model::LMM) = error("mss is not defined for $(typeof(model)).")

StatsBase.rss(model::LMM) = error("rss is not defined for $(typeof(model)).")

StatsBase.informationmatrix(model::LMM; expected::Bool = true) =
    error("informationmatrix is not defined for $(typeof(model)).")

function StatsBase.r2(model::LMM)
    Base.depwarn("The default r² method for linear models is deprecated. " *
                 "Packages should define their own methods.", :r2)
    mss(model) / deviance(model)
end

function StatsBase.r2(model::LMM, variant::Symbol)
    loglikbased = (:McFadden, :CoxSnell, :Nagelkerke)
    if variant in loglikbased
        ll = loglikelihood(model)
        ll0 = nullloglikelihood(model)
        if variant == :McFadden
            1 - ll/ll0
        elseif variant == :CoxSnell
            1 - exp(2 * (ll0 - ll) / nobs(model))
        elseif variant == :Nagelkerke
            (1 - exp(2 * (ll0 - ll) / nobs(model))) / (1 - exp(2 * ll0 / nobs(model)))
        end
    elseif variant == :devianceratio
        dev  = deviance(model)
        dev0 = nulldeviance(model)
        1 - dev/dev0
    else
        error("variant must be one of $(join(loglikbased, ", ")) or :devianceratio")
    end
end

const r² = r2

StatsBase.adjr2(model::LMM) = error("adjr2 is not defined for $(typeof(model)).")

function StatsBase.adjr2(model::LMM, variant::Symbol)
    k = dof(model)
    if variant == :McFadden
        ll = loglikelihood(model)
        ll0 = nullloglikelihood(model)
        1 - (ll - k)/ll0
    elseif variant == :devianceratio
        n = nobs(model)
        dev  = deviance(model)
        dev0 = nulldeviance(model)
        1 - (dev*(n-1))/(dev0*(n-k))
    else
        error("variant must be one of :McFadden or :devianceratio")
    end
end

const adjr² = adjr2

StatsBase.meanresponse(model::LMM) = error("meanresponse is not defined for $(typeof(model)).")
=#
