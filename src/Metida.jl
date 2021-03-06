# Metida
# Copyright © 2019-2020 Vladimir Arnautov aka PharmCat <mail@pharmcat.net>

__precompile__()
module Metida

using Distributions, LinearAlgebra, StatsBase, ForwardDiff, CategoricalArrays
using Optim, LineSearches

import MetidaBase: Tables, MetidaModel, AbstractCovarianceStructure, AbstractCovmatMethod, AbstractCovarianceType, AbstractLMMDataBlocks
using MetidaBase.StatsModels
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

function __init__()

end

function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    isdefined(Metida, Symbol("#@covstr")) && precompile(Tuple{getfield(Metida, Symbol("#@covstr")), LineNumberNode, Module, Int})

    precompile(Tuple{typeof(Metida.reml_sweep_β), Metida.LMM{Float64}, Array{Float64, 1}})

    precompile(Tuple{typeof(Metida.rholinkpsigmoid), Float64})
    precompile(Tuple{typeof(Metida.rholinkpsigmoidr), Float64})

    precompile(Tuple{typeof(Metida.varlinkvecapply!),  Array{Float64, 1},  Array{Symbol, 1}})
    precompile(Tuple{typeof(Metida.varlinkrvecapply!),  Array{Float64, 1},  Array{Symbol, 1}})
    precompile(Tuple{typeof(Metida.varlinkvecapply),  Array{Float64, 1},  Array{Symbol, 1}})

    precompile(Tuple{typeof(Metida.vlink), Float64})
    precompile(Tuple{typeof(Metida.vlinkr), Float64})

    precompile(Tuple{typeof(Metida.initvar),  Array{Float64, 1},  Array{Float64, 2}})

end
_precompile_()
#include(".jl")
end # module
