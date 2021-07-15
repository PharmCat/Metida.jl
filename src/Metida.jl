# Metida
# Copyright © 2019-2020 Vladimir Arnautov aka PharmCat <mail@pharmcat.net>

__precompile__()
module Metida

using Distributions, LinearAlgebra, StatsBase, ForwardDiff, CategoricalArrays#, LoopVectorization
using Optim, LineSearches
using MetidaBase
using StatsModels
import MetidaBase: Tables, MetidaModel, AbstractCovarianceStructure, AbstractCovmatMethod, AbstractCovarianceType, AbstractLMMDataBlocks, MetidaTable, metida_table, PrettyTables

import LinearAlgebra:checksquare
import StatsModels: @formula
import StatsBase: fit, fit!, coef, coefnames, confint, nobs, dof_residual, dof, loglikelihood, aic, bic, aicc, isfitted, vcov, stderror, modelmatrix, response
import Base:show

export @formula, @covstr,
SI, ScaledIdentity,
DIAG, Diag,
AR, Autoregressive,
ARH, HeterogeneousAutoregressive,
CS, CompoundSymmetry,
CSH, HeterogeneousCompoundSymmetry,
ARMA, AutoregressiveMovingAverage,
TOEP, Toeplitz,
TOEPP, ToeplitzParameterized,
TOEPH, HeterogeneousToeplitz,
TOEPHP, HeterogeneousToeplitzParameterized,
CovarianceType, CovmatMethod,
fit!, LMM, VarEffect, theta, logreml, m2logreml, thetalength, dof_satter, dof_contain, rankx, caic, lcontrast, anova,
gmatrix, rmatrix, vmatrix!,
AbstractCovarianceType, AbstractCovmatMethod, MetidaModel

export coef, coefnames, confint, nobs, dof_residual, dof, loglikelihood, aic, bic, aicc, isfitted, vcov, stderror, modelmatrix, response

include("sweep.jl")
include("varstruct.jl")
include("gmat.jl")
include("rmat.jl")
include("linearalgebra.jl")
include("options.jl")
include("modelresult.jl")
include("lmmdata.jl")
include("lmm.jl")
include("reml.jl")
include("ml.jl")
include("fit.jl")
include("showutils.jl")
include("statsbase.jl")
include("utils.jl")
include("dof_satter.jl")
include("dof_contain.jl")
include("fvalue.jl")
include("anova.jl")

end # module
