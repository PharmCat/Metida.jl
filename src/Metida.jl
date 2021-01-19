# Metida
# Copyright © 2019-2020 Vladimir Arnautov aka PharmCat <mail@pharmcat.net>

__precompile__()
module Metida

using Distributions, LinearAlgebra, StatsBase, StatsModels, Tables, ForwardDiff, CategoricalArrays
using Optim, LineSearches

#using TimerOutputs

import LinearAlgebra:checksquare

import StatsBase: fit, fit!, coef
import Base:show

export @formula, @covstr,
SI, ScaledIdentity,
DIAG, Diag,
AR, Autoregressive,
ARH, HeterogeneousAutoregressive,
CS, CompoundSymmetry,
CSH, HeterogeneousCompoundSymmetry,
fit!, LMM, VarEffect, theta, logreml, m2logreml, thetalength

include("abstracttype.jl")
include("sweep.jl")
include("utils.jl")
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
include("initg.jl")

function __init__()

end

function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    isdefined(Metida, Symbol("#@covstr")) && precompile(Tuple{getfield(Metida, Symbol("#@covstr")), LineNumberNode, Module, Int})
    precompile(Tuple{typeof(Metida.ffx), Int64})
    precompile(Tuple{typeof(Metida.ffxpone), Int64})

    precompile(Tuple{typeof(Metida.reml_sweep_β), Metida.LMM{Float64}, Array{Float64, 1}})
    precompile(Tuple{typeof(Metida.reml_sweep_β_b), Metida.LMM{Float64}, Array{Float64, 1}})

    precompile(Tuple{typeof(Metida.rholinkpsigmoid), Float64})

    precompile(Tuple{typeof(Metida.varlinkvecapply!), Array{Float64, 1}, Array{Function, 1}})
    precompile(Tuple{typeof(Metida.vlink), Float64})
end
_precompile_()
#include(".jl")
end # module
