using DataFrames, CSV, StatsModels, LinearAlgebra, ForwardDiff, ForwardDiff, Optim, Distributions
using NLopt
using Metida
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

################################################################################
# Metida
################################################################################

lmm = Metida.LMM(@formula(response ~1 + factor*time), ftdf;
random = Metida.VarEffect(Metida.@covstr(1 + time|subject&factor), Metida.CSH),
)
@benchmark Metida.fit!($lmm, hes = false; maxthreads = 16) seconds = 15
#@time Metida.fit!(lmm, hes = false)
#=
BenchmarkTools.Trial: 893 samples with 1 evaluation.
 Range (min … max):  14.780 ms … 31.701 ms  ┊ GC (min … max): 0.00% … 50.43%
 Time  (median):     15.356 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   16.793 ms ±  3.858 ms  ┊ GC (mean ± σ):  7.60% ± 13.35%

  ▅██▅▃▄▄▁                                             ▁▂
  ████████▇▄▇▅▄▇▇▄▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▅████▇██ █
  14.8 ms      Histogram: log(frequency) by time      28.6 ms <

 Memory estimate: 22.03 MiB, allocs estimate: 31265.

BenchmarkTools.Trial: 693 samples with 1 evaluation.
 Range (min … max):   5.497 ms … 839.792 ms  ┊ GC (min … max):  0.00% … 98.92%
 Time  (median):      9.526 ms               ┊ GC (median):     0.00%
 Time  (mean ± σ):   22.012 ms ±  90.417 ms  ┊ GC (mean ± σ):  56.52% ± 13.40%

  █
  █▃▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▃▁▃▄▁▃▃ ▆
  5.5 ms        Histogram: log(frequency) by time       652 ms <

 Memory estimate: 22.63 MiB, allocs estimate: 37225.
=#

################################################################################
# MetidaNLopt
################################################################################

lmm = Metida.LMM(@formula(response ~1 + factor*time), ftdf;
random = Metida.VarEffect(Metida.@covstr(1 + time|subject&factor), Metida.CSH),
)
@benchmark Metida.fit!($lmm, hes = false, solver = :nlopt) seconds = 15

#=
BenchmarkTools.Trial: 333 samples with 1 evaluation.
 Range (min … max):  19.021 ms … 198.344 ms  ┊ GC (min … max): 0.00% …  0.00%
 Time  (median):     42.596 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   45.055 ms ±  20.382 ms  ┊ GC (mean ± σ):  7.54% ± 12.83%

          ▁▃▄▆█▆▂
  ▄▄▅▄▄▄▅████████▆▄▂▂▁▁▁▃▂▁▂▁▁▁▃▂▃▂▃▃▁▂▂▁▂▁▁▁▁▂▁▁▂▁▁▁▁▁▁▁▁▁▂▂▂ ▃
  19 ms           Histogram: frequency by time          141 ms <

 Memory estimate: 17.83 MiB, allocs estimate: 125293.
=#
lmm = Metida.LMM(@formula(response ~1 + factor*time), ftdf;
random = Metida.VarEffect(Metida.@covstr(1 + time|factor), Metida.ARH),
)
@benchmark  Metida.fit!(lmm, solver = :nlopt, hes = false)
#=
#No @natch
julia> @benchmark  Metida.fit!(lmm, solver = :nlopt, hes = false)
BenchmarkTools.Trial: 58 samples with 1 evaluation.
 Range (min … max):  80.086 ms … 97.076 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     86.918 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   87.458 ms ±  3.918 ms  ┊ GC (mean ± σ):  2.02% ± 2.75%

     ▄               ▄  ▄▁▁▄██ ▁▁        ▄ ▁                ▁
  ▆▁▆█▁▁▁▁▆▆▁▆▆▆▁▆▁▁▁█▆▁██████▆██▆▆▆▁▁▆▁▁█▁█▁▁▁▁▆▆▆▆▆▆▆▁▁▁▁▆█ ▁
  80.1 ms         Histogram: frequency by time          95 ms <

 Memory estimate: 33.28 MiB, allocs estimate: 2137.
=#
#=
#@batcj
BenchmarkTools.Trial: 101 samples with 1 evaluation.
 Range (min … max):  43.896 ms … 66.160 ms  ┊ GC (min … max): 0.00% … 19.07%
 Time  (median):     47.827 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   49.519 ms ±  4.247 ms  ┊ GC (mean ± σ):  4.05% ±  7.55%

            █
  ▃▃▃▁▃▃▃▄████▇▆▃▃▄▁▃▃▁▃▃▁▃▁▁▁▃▁▄▁▄▃▃▄▃▃▁▁▃▁▃▁▁▁▁▁▁▁▁▃▁▁▁▁▁▁▃ ▃
  43.9 ms         Histogram: frequency by time        64.6 ms <

 Memory estimate: 33.42 MiB, allocs estimate: 4009.
=#

################################################################################
# MetidaCu
################################################################################

lmm = Metida.LMM(@formula(response ~1 + factor*time), ftdf;
random = Metida.VarEffect(Metida.@covstr(1 + time|subject&factor), Metida.CSH),
)
@benchmark Metida.fit!($lmm, hes = false, solver = :cuda) seconds = 15

#=

=#

################################################################################
################################################################################

################################################################################
# Metida
################################################################################

lmm = Metida.LMM(@formula(tumorsize ~ 1 + CancerStage), hdp;
random = Metida.VarEffect(Metida.@covstr(1|HID), Metida.DIAG),
)
@benchmark  Metida.fit!(lmm, hes = false, maxthreads = 16)

#=
BenchmarkTools.Trial: 1 sample with 1 evaluation.
 Single result which took 10.214 s (1.64% GC) to evaluate,
 with a memory estimate of 3.64 GiB, over 103755 allocations.
=#

################################################################################
# MetidaNLopt
################################################################################

lmm = Metida.LMM(@formula(tumorsize ~ 1 + CancerStage), hdp;
random = Metida.VarEffect(Metida.@covstr(1|HID), Metida.DIAG),
)
@benchmark  Metida.fit!(lmm, solver = :nlopt, hes = false)

#=
BenchmarkTools.Trial: 8 samples with 1 evaluation.
 Range (min … max):  517.141 ms … 822.903 ms  ┊ GC (min … max):  0.00% … 13.60%
 Time  (median):     644.920 ms               ┊ GC (median):    11.81%
 Time  (mean ± σ):   672.774 ms ± 115.055 ms  ┊ GC (mean ± σ):   8.02% ±  6.12%

  ▁            █          ▁ ▁                           ▁▁    ▁
  █▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁█▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁██▁▁▁▁█ ▁
  517 ms           Histogram: frequency by time          823 ms <

 Memory estimate: 919.67 MiB, allocs estimate: 39482.
=#

################################################################################
# MetidaCu
################################################################################

lmm = Metida.LMM(@formula(tumorsize ~ 1 + CancerStage), hdp;
random = Metida.VarEffect(Metida.@covstr(1|HID), Metida.DIAG),
)
@benchmark  Metida.fit!($lmm, solver = :cuda, hes = false)

#=
BenchmarkTools.Trial: 2 samples with 1 evaluation.
 Range (min … max):  3.331 s …   3.453 s  ┊ GC (min … max): 1.11% … 1.22%
 Time  (median):     3.392 s              ┊ GC (median):    1.16%
 Time  (mean ± σ):   3.392 s ± 85.816 ms  ┊ GC (mean ± σ):  1.16% ± 0.08%

  █                                                       █
  █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  3.33 s         Histogram: frequency by time        3.45 s <

 Memory estimate: 1.10 GiB, allocs estimate: 7312401
=#
