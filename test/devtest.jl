using DataFrames, CSV, StatsModels, LinearAlgebra, ForwardDiff, ForwardDiff, Optim, Distributions
using NLopt
using SnoopCompile
using LineSearches
using BenchmarkTools
path    = dirname(@__FILE__)
cd(path)
df0         = CSV.File(path*"/csv/df0.csv"; types = [String, String, String, String, Float64, Float64]) |> DataFrame
df1         = CSV.File(path*"/csv/df1.csv"; types = [String, String, String, String, Float64, Float64]) |> DataFrame
ftdf         = CSV.File(path*"/csv/1fptime.csv"; types = [String, String, Float64, Float64]) |> DataFrame
ftdf2        = CSV.File(path*"/csv/1freparma.csv"; types = [String, String, Float64, Float64]) |> DataFrame
ftdf3        = CSV.File(path*"/csv/2f2rand.csv"; types =
[String,  Float64, Float64, String, String, String, String, String]) |> DataFrame



lmm = Metida.LMM(@formula(response ~1 + factor*time), ftdf;
random = Metida.VarEffect(Metida.@covstr(1 + time|subject&factor), Metida.CSH),
)
@benchmark fit!($lmm, hes = false) seconds = 15

#=
BenchmarkTools.Trial: 527 samples with 1 evaluation.
 Range (min … max):  14.958 ms … 177.005 ms  ┊ GC (min … max):  0.00% … 89.00%
 Time  (median):     22.181 ms               ┊ GC (median):     0.00%
 Time  (mean ± σ):   28.448 ms ±  25.442 ms  ┊ GC (mean ± σ):  18.12% ± 17.00%

  ▂▄▇█▄▁
  ██████▇██▅▆▆▁▁▁▁▁▁▁▁▁▁▁▁▁▁▄▁▁▁▁▁▁▁▁▁▁▁▁▄▁▁▁▁▄▁▁▄█▄▁▆▅▅▁▄▄▁▁▄ ▇
  15 ms         Histogram: log(frequency) by time       150 ms <

 Memory estimate: 55.01 MiB, allocs estimate: 209813.
=#


lmm = Metida.LMM(@formula(tumorsize ~ 1 + CancerStage), hdp;
random = Metida.VarEffect(Metida.@covstr(1|HID), Metida.DIAG),
)
@benchmark  Metida.fit!(lmm, hes = false)

#=
BenchmarkTools.Trial: 1 sample with 1 evaluation.
 Single result which took 10.214 s (1.64% GC) to evaluate,
 with a memory estimate of 3.64 GiB, over 103755 allocations.
=#

lmm = Metida.LMM(@formula(tumorsize ~ 1 + CancerStage), hdp;
random = Metida.VarEffect(Metida.@covstr(1|HID), Metida.DIAG),
)
@benchmark  Metida.fit!(lmm, solver = :nlopt)
