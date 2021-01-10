
#=
StatsBase.coef(model::LMM) = error("coef is not defined for $(typeof(model)).")

StatsBase.coefnames(model::LMM) = error("coefnames is not defined for $(typeof(model)).")

StatsBase.coeftable(model::LMM) = error("coeftable is not defined for $(typeof(model)).")

StatsBase.confint(model::LMM) = error("confint is not defined for $(typeof(model)).")

StatsBase.deviance(model::LMM) = error("deviance is not defined for $(typeof(model)).")

StatsBase.islinear(model::LMM) = error("islinear is not defined for $(typeof(model)).")

StatsBase.nulldeviance(model::LMM) =
    error("nulldeviance is not defined for $(typeof(model)).")

StatsBase.loglikelihood(model::LMM) =
    error("loglikelihood is not defined for $(typeof(model)).")

StatsBase.nullloglikelihood(model::LMM) =
        error("nullloglikelihood is not defined for $(typeof(model)).")

StatsBase.score(model::LMM) = error("score is not defined for $(typeof(model)).")

StatsBase.nobs(model::LMM) = error("nobs is not defined for $(typeof(model)).")

StatsBase.dof(model::LMM) = error("dof is not defined for $(typeof(model)).")

StatsBase.mss(model::LMM) = error("mss is not defined for $(typeof(model)).")

StatsBase.rss(model::LMM) = error("rss is not defined for $(typeof(model)).")

StatsBase.informationmatrix(model::LMM; expected::Bool = true) =
    error("informationmatrix is not defined for $(typeof(model)).")

StatsBase.stderror(model::LMM) = sqrt.(diag(vcov(model)))

StatsBase.vcov(model::LMM) = error("vcov is not defined for $(typeof(model)).")

StatsBase.weights(model::LMM) = error("weights is not defined for $(typeof(model)).")

StatsBase.isfitted(model::LMM) = error("isfitted is not defined for $(typeof(model)).")

StatsBase.aic(model::LMM) = -2loglikelihood(model) + 2dof(model)

function StatsBase.aicc(model::LMM)
    k = dof(model)
    n = nobs(model)
    -2loglikelihood(model) + 2k + 2k*(k+1)/(n-k-1)
end

StatsBase.bic(model::LMM) = -2loglikelihood(model) + dof(model)*log(nobs(model))

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

StatsBase.response(model::LMM) = error("response is not defined for $(typeof(model)).")

StatsBase.responsename(model::LMM) = error("responsename is not defined for $(typeof(model)).")

StatsBase.meanresponse(model::LMM) = error("meanresponse is not defined for $(typeof(model)).")

StatsBase.modelmatrix(model::LMM) = error("modelmatrix is not defined for $(typeof(model)).")

StatsBase.crossmodelmatrix(model::LMM) = (x = modelmatrix(model); Symmetric(x' * x))

StatsBase.leverage(model::LMM) = error("leverage is not defined for $(typeof(model)).")

StatsBase.residuals(model::LMM) = error("residuals is not defined for $(typeof(model)).")

StatsBase.predict(model::LMM) = error("predict is not defined for $(typeof(model)).")

StatsBase.predict!(model::LMM) = error("predict! is not defined for $(typeof(model)).")

StatsBase.dof_residual(model::LMM) = error("dof_residual is not defined for $(typeof(model)).")

=#
