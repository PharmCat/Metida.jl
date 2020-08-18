# Metida
# Copyright © 2019-2020 Vladimir Arnautov aka PharmCat <mail@pharmcat.net>

__precompile__()
module Metida

using Distributions, LinearAlgebra, StatsBase, StatsModels, Tables

import Base:show

include("abstracttype.jl")
include("linearalgebra.jl")
include("lmm.jl")

#include(".jl")
end # module
