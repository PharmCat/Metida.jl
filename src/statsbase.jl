
#=

StatsBase.coeftable(model::LMM) = error("coeftable is not defined for $(typeof(model)).")

StatsBase.confint(model::LMM) = error("confint is not defined for $(typeof(model)).")

StatsBase.deviance(model::LMM) = error("deviance is not defined for $(typeof(model)).")

StatsBase.islinear(model::LMM) = error("islinear is not defined for $(typeof(model)).")

StatsBase.nulldeviance(model::LMM) =
    error("nulldeviance is not defined for $(typeof(model)).")

StatsBase.nullloglikelihood(model::LMM) =
        error("nullloglikelihood is not defined for $(typeof(model)).")

StatsBase.score(model::LMM) = error("score is not defined for $(typeof(model)).")

=#

#=
REML: n = total number of observation - number fixed effect parameters; d = number of covariance parameters
ML:, n = total number of observation; d = number of fixed effect parameters + number of covariance parameters.
=#
"""
    StatsBase.coef(lmm::LMM) = copy(lmm.result.beta)
"""
StatsBase.coef(lmm::LMM) = copy(lmm.result.beta)

"""
    StatsBase.coefnames(lmm::LMM) = StatsBase.coefnames(lmm.mf)
"""
StatsBase.coefnames(lmm::LMM) = StatsBase.coefnames(lmm.mf)

"""
    StatsBase.nobs(lmm::LMM)
"""
function StatsBase.nobs(lmm::LMM)
    return length(lmm.data.yv)
end

"""
    StatsBase.dof_residual(lmm::LMM)
"""
function StatsBase.dof_residual(lmm::LMM)
    nobs(lmm) - lmm.rankx
end

"""
    StatsBase.dof(lmm::LMM)
"""
function StatsBase.dof(lmm::LMM)
    lmm.nfixed + lmm.covstr.tl
end

"""
    StatsBase.loglikelihood(lmm::LMM)
"""
function StatsBase.loglikelihood(lmm::LMM)
    -lmm.result.reml/2
end

"""
    StatsBase.aic(lmm::LMM)
"""
function StatsBase.aic(lmm::LMM)
    l = loglikelihood(lmm)
    d = lmm.covstr.tl
    -2l + 2d
end

"""
    StatsBase.bic(lmm::LMM)
"""
function StatsBase.bic(lmm::LMM)
    l = loglikelihood(lmm)
    d = lmm.covstr.tl
    n = nobs(lmm) - lmm.nfixed
    -2l + d * log(n)
end

"""
    StatsBase.aicc(lmm::LMM)
"""
function StatsBase.aicc(lmm::LMM)
    l = loglikelihood(lmm)
    d = lmm.covstr.tl
    n = nobs(lmm) - lmm.nfixed
    -2l + (2d * n) / (n - d - 1.0)
end

"""
    caic(lmm::LMM)
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

StatsBase.vcov(lmm::LMM) = copy(lmm.result.c)

StatsBase.stderror(lmm::LMM) = sqrt.(diag(vcov(lmm)))

StatsBase.modelmatrix(lmm::LMM) = lmm.data.xv

StatsBase.response(lmm::LMM) = lmm.data.yv
#=
StatsBase.mss(model::LMM) = error("mss is not defined for $(typeof(model)).")

StatsBase.rss(model::LMM) = error("rss is not defined for $(typeof(model)).")

StatsBase.informationmatrix(model::LMM; expected::Bool = true) =
    error("informationmatrix is not defined for $(typeof(model)).")

StatsBase.weights(model::LMM) = error("weights is not defined for $(typeof(model)).")

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

StatsBase.responsename(model::LMM) = error("responsename is not defined for $(typeof(model)).")

StatsBase.meanresponse(model::LMM) = error("meanresponse is not defined for $(typeof(model)).")

StatsBase.crossmodelmatrix(model::LMM) = (x = modelmatrix(model); Symmetric(x' * x))

StatsBase.leverage(model::LMM) = error("leverage is not defined for $(typeof(model)).")

StatsBase.residuals(model::LMM) = error("residuals is not defined for $(typeof(model)).")

StatsBase.predict(model::LMM) = error("predict is not defined for $(typeof(model)).")

StatsBase.predict!(model::LMM) = error("predict! is not defined for $(typeof(model)).")


=#