# Metida
# Copyright © 2019-2020 Vladimir Arnautov aka PharmCat <mail@pharmcat.net>

__precompile__()
module Metida

using Distributions, LinearAlgebra, StatsBase, StatsModels, Tables, ForwardDiff, BlockDiagonals, BlockArrays#, SweepOperator
using Optim

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
include("reml.jl")
include("ml.jl")
include("lmmdata.jl")
include("lmm.jl")
include("optfgh.jl")
include("fit.jl")



#include(".jl")
end # module
