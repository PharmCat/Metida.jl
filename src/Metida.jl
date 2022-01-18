# Metida
# Copyright © 2019-2020 Vladimir Arnautov aka PharmCat <mail@pharmcat.net>

__precompile__()
module Metida

using Distributions, LinearAlgebra, StatsBase, ForwardDiff, CategoricalArrays, Random, Optim, LineSearches, MetidaBase#, SparseArrays#, Polyester#, LoopVectorization
using ProgressMeter
using StatsModels
import MetidaBase: Tables, MetidaModel, AbstractCovarianceStructure, AbstractCovmatMethod, AbstractCovarianceType, AbstractLMMDataBlocks, MetidaTable, metida_table, PrettyTables#, indsdict!
import MetidaBase.PrettyTables: TextFormat, pretty_table, tf_borderless, ft_printf
import LinearAlgebra:checksquare, BlasFloat
import StatsModels: @formula, termvars, ModelFrame
import StatsBase: fit, fit!, coef, coefnames, confint, nobs, dof_residual, dof, loglikelihood, aic, bic, aicc, isfitted, vcov, stderror, modelmatrix, response, CoefTable, coeftable
import Base:show, rand, ht_keyindex
import Random: default_rng, AbstractRNG, rand!

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
SPEXP, SpatialExponential,
SPPOW, SpatialPower,
SPGAU, SpatialGaussian,
CovarianceType, CovmatMethod,
fit!, LMM, VarEffect, theta, logreml, m2logreml, thetalength, dof_satter, dof_contain, rankx, caic, lcontrast, typeiii, estimate, contrast,
gmatrix, rmatrix, vmatrix!,
AbstractCovarianceType, AbstractCovmatMethod, MetidaModel,
getlog, rand, rand!,
bootstrap

export coef, coefnames, confint, nobs, dof_residual, dof, loglikelihood, aic, bic, aicc, isfitted, vcov, stderror, modelmatrix, response

num_cores() = Int(MetidaBase.num_cores())

const LDCORR = sqrt(eps())
const LOGLDCORR = log(sqrt(eps()))
const NEWTON_OM = Optim.Newton(;alphaguess = LineSearches.InitialHagerZhang(), linesearch = LineSearches.HagerZhang())
const LBFGS_OM  = Optim.LBFGS(;alphaguess = LineSearches.InitialStatic(), linesearch = LineSearches.Static())

const METIDA_SETTINGS = Dict(:MAX_THREADS => num_cores())

include("exceptions.jl")
include("sweep.jl")
include("vartypes.jl")
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
include("typeiii.jl")
include("estimate.jl")
include("random.jl")
include("miboot.jl")

    const NOREPEAT = VarEffect(Metida.@covstr(1|1), Metida.ScaledIdentity())
end # module
