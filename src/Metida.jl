# Metida
# Copyright © 2019-2020 Vladimir Arnautov aka PharmCat <mail@pharmcat.net>

__precompile__()
module Metida

using Distributions, LinearAlgebra, StatsBase, StatsModels, Tables, ForwardDiff, BlockDiagonals, BlockArrays#, SweepOperator
using Optim
#using NLopt

import LinearAlgebra:checksquare

import StatsBase: fit, fit!
import Base:show

export covstr, VC

include("abstracttype.jl")
include("sweep.jl")
include("utils.jl")
include("varstruct.jl")
include("linearalgebra.jl")
include("covmat.jl")
include("matderiv.jl")
include("lmmdata.jl")
include("lmm.jl")
include("reml.jl")
include("ml.jl")
include("optfgh.jl")
include("modelresult.jl")
include("fit.jl")

function __init__()
    #a  = ones(Float64, 2, 2)
    #ve = VarEffect(@covstr(formulation), CSH)
    #G  = gmat([0.5, 0.4, 0.1], 2, ve)
    #ve = VarEffect(@covstr(formulation), VC)
    #G  = gmat([0.5, 0.4], 2, ve)
end
function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    isdefined(NLSolversBase, Symbol("#42#48")) && Base.precompile(Tuple{getfield(NLSolversBase, Symbol("#42#48")),Array{Float64,1},Array{Float64,1}})
    isdefined(NLSolversBase, Symbol("#43#49")) && Base.precompile(Tuple{getfield(NLSolversBase, Symbol("#43#49")),Array{Float64,2},Array{Float64,1}})
end
_precompile_()
#include(".jl")
end # module
