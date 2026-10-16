# Инициализация окружения для отдельных секций валидации.
using Metida
using  Test, CSV, DataFrames, StatsBase, PrettyTables, Distributions, LinearAlgebra, CategoricalArrays, Random, StableRNGs
path    = dirname(@__FILE__)

# MixedModels 
const MM_AVAILABLE = try
    @eval using MixedModels
    true
catch err
    @warn "MixedModels недоступна: категория 1 выполнена не будет" exception = err
    false
end
include(joinpath(path, "testdata.jl"))

# 
isdefined(@__MODULE__, :REFVALUES) || include(joinpath(path, "references.jl"))
