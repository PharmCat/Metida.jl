# Metida
# Copyright © 2019-2020 Vladimir Arnautov aka PharmCat <mail@pharmcat.net>

__precompile__()
module Metida

using Distributions, LinearAlgebra, StatsBase, ForwardDiff, CategoricalArrays, Polyester#, LoopVectorization
using Optim, LineSearches, MetidaBase
using StatsModels
import MetidaBase: Tables, MetidaModel, AbstractCovarianceStructure, AbstractCovmatMethod, AbstractCovarianceType, AbstractLMMDataBlocks, MetidaTable, metida_table, PrettyTables

import LinearAlgebra:checksquare
import StatsModels: @formula, termvars
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
fit!, LMM, VarEffect, theta, logreml, m2logreml, thetalength, dof_satter, dof_contain, rankx, caic, lcontrast, typeiii, estimate, contrast,
gmatrix, rmatrix, vmatrix!,
AbstractCovarianceType, AbstractCovmatMethod, MetidaModel,
getlog

export coef, coefnames, confint, nobs, dof_residual, dof, loglikelihood, aic, bic, aicc, isfitted, vcov, stderror, modelmatrix, response

const LDCORR = sqrt(eps())
const LOGLDCORR = log(sqrt(eps()))
const NEWTON_OM = Optim.Newton(;alphaguess = LineSearches.InitialHagerZhang(), linesearch = LineSearches.HagerZhang())
const LBFGS_OM  = Optim.LBFGS(;alphaguess = LineSearches.InitialStatic(), linesearch = LineSearches.Static())

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
include("typeiii.jl")
include("estimate.jl")


function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(confint)),NamedTuple{(:ddf,), Tuple{Symbol}},typeof(confint),LMM{Float64}})
    Base.precompile(Tuple{Core.kwftype(typeof(estimate)),NamedTuple{(:level,), Tuple{Float64}},typeof(estimate),LMM{Float64},Vector{Int64}})
    Base.precompile(Tuple{Core.kwftype(typeof(fit!)),NamedTuple{(:aifirst, :init), Tuple{Bool, Vector{Float64}}},typeof(fit!),LMM{Float64}})
    Base.precompile(Tuple{Core.kwftype(typeof(fit!)),NamedTuple{(:aifirst,), Tuple{Bool}},typeof(fit!),LMM{Float64}})
    Base.precompile(Tuple{Core.kwftype(typeof(fit!)),NamedTuple{(:aifirst,), Tuple{Symbol}},typeof(fit!),LMM{Float64}})
    Base.precompile(Tuple{Core.kwftype(typeof(fit!)),NamedTuple{(:hes,), Tuple{Bool}},typeof(fit!),LMM{Float64}})
    Base.precompile(Tuple{Core.kwftype(typeof(fit!)),NamedTuple{(:init,), Tuple{Vector{Float64}}},typeof(fit!),LMM{Float64}})
    Base.precompile(Tuple{Core.kwftype(typeof(fit!)),NamedTuple{(:rholinkf,), Tuple{Symbol}},typeof(fit!),LMM{Float64}})
    Base.precompile(Tuple{Core.kwftype(typeof(fit!)),NamedTuple{(:varlinkf,), Tuple{Symbol}},typeof(fit!),LMM{Float64}})
    Base.precompile(Tuple{Core.kwftype(typeof(fit!)),NamedTuple{(:verbose, :io), Tuple{Int64, IOBuffer}},typeof(fit!),LMM{Float64}})
    Base.precompile(Tuple{Core.kwftype(typeof(typeiii)),NamedTuple{(:ddf,), Tuple{Symbol}},typeof(typeiii),LMM{Float64}})
    Base.precompile(Tuple{Core.kwftype(typeof(varlinkrvecapply!)),NamedTuple{(:varlinkf, :rholinkf), Tuple{Symbol, Symbol}},typeof(varlinkrvecapply!),Vector{Float64},Vector{Symbol}})
    Base.precompile(Tuple{Core.kwftype(typeof(varlinkvecapply!)),NamedTuple{(:varlinkf, :rholinkf), Tuple{Symbol, Symbol}},typeof(varlinkvecapply!),Vector{Float64},Vector{Symbol}})
    Base.precompile(Tuple{Core.kwftype(typeof(varlinkvecapply)),NamedTuple{(:varlinkf, :rholinkf), Tuple{Symbol, Symbol}},typeof(varlinkvecapply),Vector{Float64},Vector{Symbol}})

    Base.precompile(Tuple{Type{CovmatMethod},Function,Function})
    Base.precompile(Tuple{typeof(TOEPHP),Int64})
    Base.precompile(Tuple{typeof(TOEPP),Int64})
    Base.precompile(Tuple{typeof(aicc),LMM{Float64}})
    Base.precompile(Tuple{typeof(coefnames),LMM{Float64}})
    Base.precompile(Tuple{typeof(confint),LMM{Float64}})
    Base.precompile(Tuple{typeof(contrast),LMM{Float64},Matrix{Int64}})
    Base.precompile(Tuple{typeof(dof_contain),LMM{Float64},Int64})
    Base.precompile(Tuple{typeof(dof_contain_f),LMM{Float64},Int64})
    Base.precompile(Tuple{typeof(dof_satter),LMM{Float64},Int64})
    Base.precompile(Tuple{typeof(dof_satter),LMM{Float64},Matrix{Float64}})
    Base.precompile(Tuple{typeof(dof_satter),LMM{Float64},Matrix{Int64}})
    Base.precompile(Tuple{typeof(dof_satter),LMM{Float64}})

    Base.precompile(Tuple{typeof(fit!),LMM{Float64}})
    Base.precompile(Tuple{typeof(fulldummycodingdict),InteractionTerm{Tuple{Term, Term}}})
    Base.precompile(Tuple{typeof(fvalue),LMM{Float64},Matrix{Float64}})
    Base.precompile(Tuple{typeof(fvalue),LMM{Float64},Matrix{Int64}})

    Base.precompile(Tuple{typeof(get_symb),InterceptTerm{true}})
    Base.precompile(Tuple{typeof(get_symb),Tuple{Term, Term}})
    Base.precompile(Tuple{typeof(gradc),LMM{Float64},Vector{Float64}})
    Base.precompile(Tuple{typeof(hessian),LMM{Float64},Vector{Float64}})
    Base.precompile(Tuple{typeof(initvar),Vector{Float64},Matrix{Float64}})
    Base.precompile(Tuple{typeof(lmmlog!),Base.TTY,LMM{Float64},Int64,LMMLogMsg})
    Base.precompile(Tuple{typeof(lmmlog!),IOBuffer,LMM{Float64},Int64,LMMLogMsg})
    Base.precompile(Tuple{typeof(logreml),LMM{Float64}})
    Base.precompile(Tuple{typeof(m2logreml),LMM{Float64},Vector{Float64}})
    Base.precompile(Tuple{typeof(mulαtβα),Vector{Float64},Matrix{Float64}})
    Base.precompile(Tuple{typeof(mulαtβα),Vector{Int64},Matrix{Float64}})
    Base.precompile(Tuple{typeof(printmatrix),IOBuffer,Matrix{Any}})
    Base.precompile(Tuple{typeof(rmat_base_inc_b!),SubArray{Float64, 2, Matrix{Float64}, Tuple{Vector{UInt32}, Vector{UInt32}}, false},Vector{Float64},SubArray{Float64, 2, Matrix{Float64}, Tuple{Vector{UInt32}, Base.Slice{Base.OneTo{Int64}}}, false},CovStructure{Float64}})
    Base.precompile(Tuple{typeof(rmatrix),LMM{Float64},Int64})
    Base.precompile(Tuple{typeof(show),IOBuffer,ContrastTable})
    Base.precompile(Tuple{typeof(show),IOBuffer,CovStructure{Float64}})
    Base.precompile(Tuple{typeof(show),IOBuffer,LMM{Float64}})
    Base.precompile(Tuple{typeof(show),IOBuffer,ModelResult})
    Base.precompile(Tuple{typeof(theta),LMM{Float64}})
    Base.precompile(Tuple{typeof(vmatrix),LMM{Float64},Int64})
    Base.precompile(Tuple{typeof(vmatrix),Vector{Float64},LMM{Float64},Int64})

    isdefined(Metida, Symbol("#hfunc!#52")) && Base.precompile(Tuple{getfield(Metida, Symbol("#hfunc!#52")),Matrix{Float64},Vector{Float64}})
end
    _precompile_()
    const NOREPEAT = VarEffect(Metida.@covstr(1|1), Metida.ScaledIdentity())
end # module
