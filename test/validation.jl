using Metida
using  Test, CSV, DataFrames, StatsBase, PrettyTables, MixedModels
path    = dirname(@__FILE__)
include(joinpath(path, "validation_s1.jl"))
include(joinpath(path, "validation_s2.jl"))
include(joinpath(path, "validation_s3.jl"))
