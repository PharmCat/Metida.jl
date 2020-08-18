# Metida

using  Test, CSV, DataFrames, StatsModels

path    = dirname(@__FILE__)
include("testdata.jl")

@testset "  Basic test                         " begin
    lmm = Metida.LMM(@formula(var ~ sequence + period + formulation), df0)
    @test true
end
