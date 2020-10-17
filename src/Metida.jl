# Metida
# Copyright © 2019-2020 Vladimir Arnautov aka PharmCat <mail@pharmcat.net>

__precompile__()
module Metida

using Distributions, LinearAlgebra, StatsBase, StatsModels, Tables, ForwardDiff#, BlockDiagonals, BlockArrays#, SweepOperator
using Optim
#using NLopt

import LinearAlgebra:checksquare

import StatsBase: fit, fit!
import Base:show

export @formula, @covstr, VC, VarianceComponents, CSH, HeterogeneousCompoundSymmetry, SI, ScaledIdentity, fit!, LMM, VarEffect

include("abstracttype.jl")
include("sweep.jl")
include("utils.jl")
include("varstruct.jl")
include("linearalgebra.jl")
include("covmat.jl")
include("matderiv.jl")
include("options.jl")
include("modelresult.jl")
include("lmmdata.jl")
include("lmm.jl")
include("reml.jl")
include("ml.jl")
include("optfgh.jl")
include("fit.jl")
include("showutils.jl")

function __init__()

end

function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    isdefined(Metida, Symbol("#@covstr")) && precompile(Tuple{getfield(Metida, Symbol("#@covstr")), LineNumberNode, Module, Int})
    precompile(Tuple{typeof(Metida.ffx), Int64})
    precompile(Tuple{typeof(Metida.ffxpone), Int64})
    #precompile(Tuple{typeof(Metida.rcoefnames), Tuple{StatsModels.CategoricalTerm{StatsModels.FullDummyCoding, Int64, 2}, StatsModels.CategoricalTerm{StatsModels.FullDummyCoding, Int64, 4}, StatsModels.InteractionTerm{Tuple{StatsModels.CategoricalTerm{StatsModels.FullDummyCoding, Int64, 2}, StatsModels.CategoricalTerm{StatsModels.FullDummyCoding, Int64, 4}}}}, Int64, Metida.VarEffect{Metida.HeterogeneousCompoundSymmetry}})
    #precompile(Tuple{typeof(Metida.rcoefnames), Tuple{StatsModels.CategoricalTerm{StatsModels.FullDummyCoding, Int64, 2}, StatsModels.CategoricalTerm{StatsModels.FullDummyCoding, Int64, 4}}, Int64, Metida.VarEffect{Metida.HeterogeneousCompoundSymmetry}})
    #precompile(Tuple{typeof(Metida.rcoefnames), Tuple{StatsModels.CategoricalTerm{StatsModels.FullDummyCoding, Int64, 2}}, Int64, Metida.VarEffect{Metida.HeterogeneousCompoundSymmetry}})
    #precompile(Tuple{typeof(Metida.rcoefnames), Tuple{StatsModels.CategoricalTerm{StatsModels.FullDummyCoding, Int64, 2}}, Int64, Metida.VarEffect{Metida.VarianceComponents}})
    #precompile(Tuple{typeof(Metida.rcoefnames), Tuple{StatsModels.CategoricalTerm{StatsModels.FullDummyCoding, Int64, 4}}, Int64, Metida.VarEffect{Metida.VarianceComponents}})
    #precompile(Tuple{typeof(Metida.rcoefnames), Tuple{StatsModels.InteractionTerm{Tuple{StatsModels.CategoricalTerm{StatsModels.FullDummyCoding, Int64, 2}, StatsModels.CategoricalTerm{StatsModels.FullDummyCoding, Int64, 4}}}}, Int64, Metida.VarEffect{Metida.HeterogeneousCompoundSymmetry}})
    precompile(Tuple{typeof(Metida.reml_sweep_β), Metida.LMM{Float64}, Array{Float64, 1}})
    precompile(Tuple{typeof(Metida.rholinkpsigmoid), Float64})
    #precompile(Tuple{typeof(Metida.subjblocks), DataFrames.DataFrame, Symbol, Array{Float64, 2}, Array{Float64, 2}, Array{Float64, 1}, Array{Float64, 2}})
    #precompile(Tuple{typeof(Metida.subjblocks), DataFrames.DataFrame, Symbol, Array{Float64, 2}, Array{Float64, 2}, Array{Float64, 1}, Nothing})
    precompile(Tuple{typeof(Metida.varlinkvecapply!), Array{Float64, 1}, Array{Function, 1}})
    precompile(Tuple{typeof(Metida.vlink), Float64})
end
_precompile_()
#include(".jl")
end # module