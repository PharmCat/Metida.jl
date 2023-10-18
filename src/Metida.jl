# Metida
# Copyright © 2019-2020 Vladimir Arnautov aka PharmCat <mail@pharmcat.net>

__precompile__()
module Metida

using ProgressMeter, LinearAlgebra, ForwardDiff, DiffResults, Random, Optim, LineSearches, MetidaBase#, SparseArrays#, Polyester#, LoopVectorization
import StatsBase, StatsModels, Distributions

import MetidaBase: CategoricalArrays, Tables, MetidaModel, AbstractCovarianceStructure, AbstractCovmatMethod, AbstractCovarianceType, AbstractLMMDataBlocks, MetidaTable, metida_table, PrettyTables, indsdict!
import MetidaBase.CategoricalArrays: CategoricalArray, AbstractCategoricalVector
import MetidaBase.PrettyTables: TextFormat, pretty_table, tf_borderless, ft_printf

import Distributions: Normal, TDist, FDist, Chisq, MvNormal, FullNormal, ccdf, cdf, quantile
import LinearAlgebra: checksquare, BlasFloat
import StatsModels: @formula, termvars, ModelFrame, FunctionTerm, AbstractTerm, CategoricalTerm, AbstractContrasts, ConstantTerm, InterceptTerm, Term, InteractionTerm, FormulaTerm, ModelMatrix, schema, apply_schema, MatrixTerm, modelcols
import StatsBase: fit, fit!, coef, coefnames, confint, nobs, dof_residual, dof, loglikelihood, aic, bic, aicc, isfitted, vcov, mean, var, stderror, modelmatrix, response, responsename, CoefTable, coeftable, crossmodelmatrix
import Base: show, rand, ht_keyindex, getproperty
import Random: default_rng, AbstractRNG, rand!

export @formula, @covstr, @lmmformula,
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
UN, Unstructured,
CovarianceType,
fit, fit!, LMM, VarEffect, theta, logreml, m2logreml, thetalength, dof_satter, dof_contain, rankx, caic, lcontrast, typeiii, estimate, contrast,
gmatrix, rmatrix, vmatrix!, responsename, nblocks, raneff,
AbstractCovarianceType, AbstractCovmatMethod, MetidaModel,
getlog, rand, rand!,
bootstrap

export coef, coefnames, coeftable, crossmodelmatrix, confint, nobs, dof_residual, dof, loglikelihood, aic, bic, aicc, isfitted, vcov, stderror, modelmatrix, response

num_cores() = Int(MetidaBase.num_cores())

const LDCORR = sqrt(eps())
const LOGLDCORR = log(sqrt(eps()))
const NEWTON_OM = Optim.Newton(;alphaguess = LineSearches.InitialHagerZhang(), linesearch = LineSearches.HagerZhang())
const LBFGS_OM  = Optim.LBFGS(;alphaguess = LineSearches.InitialStatic(), linesearch = LineSearches.Static())
const BFGS_OM   = Optim.BFGS()
const CG_OM     = Optim.ConjugateGradient()
const NM_OM     = Optim.NelderMead()

#const METIDA_SETTINGS = Dict(:MAX_THREADS => num_cores())

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
include("lmmformula.jl")
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
