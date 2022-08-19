#using NLopt
using Metida

using DataFrames, CSV, StatsModels, LinearAlgebra, ForwardDiff, ForwardDiff, Optim, Distributions, CategoricalArrays
#using SnoopCompile
#using LineSearches
using BenchmarkTools
path    = dirname(@__FILE__)
cd(path)
df0         = CSV.File(path*"/csv/df0.csv"; types = [String, String, String, String, Float64, Float64]) |> DataFrame
df1         = CSV.File(path*"/csv/df1.csv"; types = [String, String, String, String, Float64, Float64]) |> DataFrame
ftdf         = CSV.File(path*"/csv/1fptime.csv"; types = [String, String, Float64, Float64]) |> DataFrame
ftdf2        = CSV.File(path*"/csv/1freparma.csv"; types = [String, String, Float64, Float64]) |> DataFrame
ftdf3        = CSV.File(path*"/csv/ftdf3.csv"; types =
[String,  Float64, Float64, String, String, String, String, String, Float64]) |> DataFrame
pkgversion(m::Module) = Pkg.TOML.parsefile(joinpath(dirname(string(first(methods(m.eval)).file)), "..", "Project.toml"))["version"]
# MODEL 1
################################################################################
# Metida
################################################################################
lmm = Metida.LMM(@formula(response ~1 + factor*time), ftdf;
random = Metida.VarEffect(Metida.@covstr(1 + time|subject&factor), Metida.CSH),
)
b11 = @benchmark Metida.fit!($lmm, hes = false; maxthreads = 16) seconds = 15
#@time Metida.fit!(lmm, hes = false)


################################################################################
# MetidaNLopt
################################################################################
using MetidaNLopt
b12 = @benchmark Metida.fit!($lmm, hes = false, solver = :nlopt) seconds = 15

################################################################################
# MetidaCu
################################################################################
using MetidaCu
b13 = @benchmark Metida.fit!($lmm, hes = false, solver = :cuda) seconds = 15


# MODEL 2

lmm = Metida.LMM(@formula(response ~1 + factor*time), ftdf;
random = Metida.VarEffect(Metida.@covstr(1 + time|factor), Metida.ARH),
)
b21 = @benchmark Metida.fit!($lmm, hes = false; maxthreads = 16) seconds = 15
b22 = @benchmark Metida.fit!($lmm, hes = false, solver = :nlopt) seconds = 15
b23 = @benchmark Metida.fit!($lmm, hes = false, solver = :cuda) seconds = 15

# MODEL 3

lmm = Metida.LMM(@formula(response ~1 + factor*time), ftdf;
random = Metida.VarEffect(Metida.@covstr(1 + time|subject&factor), Metida.CSH),
)

b31 = @benchmark Metida.fit!($lmm, hes = false; maxthreads = 16) seconds = 15
b32 = @benchmark Metida.fit!($lmm, hes = false, solver = :nlopt) seconds = 15
b33 = @benchmark Metida.fit!($lmm, hes = false, solver = :cuda) seconds = 15

# MODEL 4

hdp         = CSV.File("hdp.csv") |> DataFrame
transform!(hdp, :DID => categorical);
transform!(hdp, :HID=> categorical);
transform!(hdp, :Sex=> categorical);
transform!(hdp, :School=> categorical);
transform!(hdp, :pain=> categorical);
################################################################################
# Metida
################################################################################

lmm = Metida.LMM(@formula(tumorsize ~ 1 + CancerStage), hdp;
random = Metida.VarEffect(Metida.@covstr(1|HID), Metida.DIAG),
)

b41 = @benchmark Metida.fit!($lmm, hes = false; maxthreads = 16) seconds = 15
b42 = @benchmark Metida.fit!($lmm, hes = false, solver = :nlopt) seconds = 15
b43 = @benchmark Metida.fit!($lmm, hes = false, solver = :cuda) seconds = 15

# MODEL 5 maximum 1437 observation-per-subject (10 subjects)
lmm = Metida.LMM(@formula(tumorsize ~ 1 + CancerStage), hdp;
random = Metida.VarEffect(Metida.@covstr(1|ntumors), Metida.SI),
)

#b51 = @benchmark Metida.fit!($lmm, hes = false; maxthreads = 16) seconds = 15
b52 = @benchmark Metida.fit!($lmm, hes = false, solver = :nlopt) seconds = 15
b53 = @benchmark Metida.fit!($lmm, hes = false, solver = :cuda) seconds = 15

# MODEL 6: maximum 3409 observation-per-subject (4 subjects)


println("Metida version: ", pkgversion(Metida))
println("MetidaNLopt version: ", pkgversion(MetidaNLopt))
println("MetidaCu version: ", pkgversion(MetidaCu))

println("MODEL 1")
println("# Metida")
display(b11)
println("# MetidaNLopt")
display(b12)
println("# MetidaCu")
display(b13)
println()
println()

println("MODEL 2")
println("# Metida")
display(b21)
println("# MetidaNLopt")
display(b22)
println("# MetidaCu")
display(b23)
println()
println()

println("MODEL 3")
println("# Metida")
display(b31)
println("# MetidaNLopt")
display(b32)
println("# MetidaCu")
display(b33)
println()
println()

println("MODEL 4")
println("# Metida")
display(b41)
println("# MetidaNLopt")
display(b42)
println("# MetidaCu")
display(b43)
println()
println()

println("MODEL 5")
#println("# Metida")
#display(b51)
println("# MetidaNLopt")
display(b52)
println("# MetidaCu")
display(b53)
println()
println()

#Julia 1.6.3
#=
julia>
Metida version: 0.12.0
MetidaNLopt version: 0.4.0
MetidaCu version: 0.4.1
MODEL 1
# Metida
BenchmarkTools.Trial: 1184 samples with 1 evaluation.
 Range (min … max):   5.489 ms … 161.314 ms  ┊ GC (min … max):  0.00% … 94.86%
 Time  (median):      8.535 ms               ┊ GC (median):     0.00%
 Time  (mean ± σ):   12.700 ms ±  22.994 ms  ┊ GC (mean ± σ):  33.38% ± 16.90%

  ▇█▅
  ███▄▄▄▁▁▁▁▁▁▁▁▁▁▁▄▁▁▄▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▄▄▁▄▄▁▅▄▆▁▄▄▄▄▅▄▆▄ █
  5.49 ms       Histogram: log(frequency) by time       144 ms <

 Memory estimate: 22.63 MiB, allocs estimate: 37225.
# MetidaNLopt
BenchmarkTools.Trial: 173 samples with 1 evaluation.
 Range (min … max):  74.294 ms … 186.791 ms  ┊ GC (min … max): 0.00% … 58.09%
 Time  (median):     78.964 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   86.794 ms ±  22.516 ms  ┊ GC (mean ± σ):  9.19% ± 14.57%

    █▃
  ▅▆██▆▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▅█▁▁▁▁▁▁▄▁▁▄▁▄▁▄▁▁▁▁▁▁▁▄▁▁▁▄▁▄▁▄▄▁▁▄▁▄ ▄
  74.3 ms       Histogram: log(frequency) by time       176 ms <

 Memory estimate: 56.61 MiB, allocs estimate: 481209.
# MetidaCu
BenchmarkTools.Trial: 2 samples with 1 evaluation.
 Range (min … max):  9.128 s …    9.323 s  ┊ GC (min … max): 0.00% … 0.58%
 Time  (median):     9.226 s               ┊ GC (median):    0.29%
 Time  (mean ± σ):   9.226 s ± 137.689 ms  ┊ GC (mean ± σ):  0.29% ± 0.41%

  █                                                        █
  █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  9.13 s         Histogram: frequency by time         9.32 s <

 Memory estimate: 143.46 MiB, allocs estimate: 2524366.


MODEL 2
# Metida
BenchmarkTools.Trial: 18 samples with 1 evaluation.
 Range (min … max):  829.008 ms … 850.212 ms  ┊ GC (min … max): 0.00% … 1.35%
 Time  (median):     839.290 ms               ┊ GC (median):    0.58%
 Time  (mean ± σ):   839.688 ms ±   7.197 ms  ┊ GC (mean ± σ):  0.62% ± 0.63%

  ▁   █    ▁   ▁▁  ▁   █               ▁       ▁█▁▁      ▁  ▁ ▁
  █▁▁▁█▁▁▁▁█▁▁▁██▁▁█▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁████▁▁▁▁▁▁█▁▁█▁█ ▁
  829 ms           Histogram: frequency by time          850 ms <

 Memory estimate: 140.68 MiB, allocs estimate: 7919.
# MetidaNLopt
BenchmarkTools.Trial: 135 samples with 1 evaluation.
 Range (min … max):  108.482 ms … 119.274 ms  ┊ GC (min … max): 0.00% … 4.83%
 Time  (median):     110.394 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   111.933 ms ±   2.719 ms  ┊ GC (mean ± σ):  1.80% ± 2.26%

        ▂ ▂ █▂▄▃                                 ▂▂
  █▅▆▃▅▅███▇█████▆▅▁▃▁▃▃▃▁▁▁▁▃▁▁▁▁▁▃▁▁▁▁▃▃▃▅▁▆▃▇▇██▇▇▇▅▅▁▆▁▁▁▃▃ ▃
  108 ms           Histogram: frequency by time          117 ms <

 Memory estimate: 91.48 MiB, allocs estimate: 33646.
# MetidaCu
BenchmarkTools.Trial: 29 samples with 1 evaluation.
 Range (min … max):  507.681 ms … 536.903 ms  ┊ GC (min … max): 0.00% … 1.58%
 Time  (median):     518.749 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   521.751 ms ±  10.164 ms  ┊ GC (mean ± σ):  0.58% ± 0.74%

           █             ▃                          ▃     ▃   ▃
  ▇▁▁▁▇▇▁▇▁█▇▇▇▇▁▇▁▁▁▇▇▁▁█▁▇▁▇▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▇▁█▁▇▇▁▇█▇▁▁█ ▁
  508 ms           Histogram: frequency by time          537 ms <

 Memory estimate: 93.94 MiB, allocs estimate: 95784.


MODEL 3
# Metida
BenchmarkTools.Trial: 1100 samples with 1 evaluation.
 Range (min … max):   5.554 ms … 213.611 ms  ┊ GC (min … max):  0.00% … 95.95%
 Time  (median):      8.962 ms               ┊ GC (median):     0.00%
 Time  (mean ± σ):   13.684 ms ±  28.997 ms  ┊ GC (mean ± σ):  37.50% ± 16.47%

  ▇█
  ██▄▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▄▄▆▄▁▆▄▄▁▄▄▄▆ ▇
  5.55 ms       Histogram: log(frequency) by time       181 ms <

 Memory estimate: 22.63 MiB, allocs estimate: 37224.
# MetidaNLopt
BenchmarkTools.Trial: 174 samples with 1 evaluation.
 Range (min … max):  75.122 ms … 186.340 ms  ┊ GC (min … max): 0.00% … 57.72%
 Time  (median):     79.034 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   86.617 ms ±  22.856 ms  ┊ GC (mean ± σ):  8.85% ± 14.37%

   ▁█
  ▇██▇▄▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▄▇▆▁▄▁▁▁▅▄▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▄▁▅▁▁▄▁▆ ▄
  75.1 ms       Histogram: log(frequency) by time       175 ms <

 Memory estimate: 56.61 MiB, allocs estimate: 481212.
# MetidaCu
BenchmarkTools.Trial: 2 samples with 1 evaluation.
 Range (min … max):  9.221 s …   9.255 s  ┊ GC (min … max): 0.00% … 0.29%
 Time  (median):     9.238 s              ┊ GC (median):    0.14%
 Time  (mean ± σ):   9.238 s ± 24.138 ms  ┊ GC (mean ± σ):  0.14% ± 0.20%

  █                                                       █
  █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  9.22 s         Histogram: frequency by time        9.26 s <

 Memory estimate: 143.47 MiB, allocs estimate: 2524496.


MODEL 4
# Metida
BenchmarkTools.Trial: 3 samples with 1 evaluation.
 Range (min … max):  6.738 s …   6.852 s  ┊ GC (min … max): 1.36% … 0.52%
 Time  (median):     6.745 s              ┊ GC (median):    1.09%
 Time  (mean ± σ):   6.779 s ± 63.739 ms  ┊ GC (mean ± σ):  0.99% ± 0.43%

  █  █                                                    █
  █▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  6.74 s         Histogram: frequency by time        6.85 s <

 Memory estimate: 2.33 GiB, allocs estimate: 41657.
# MetidaNLopt
BenchmarkTools.Trial: 11 samples with 1 evaluation.
 Range (min … max):  1.317 s …   1.438 s  ┊ GC (min … max): 3.19% … 8.44%
 Time  (median):     1.365 s              ┊ GC (median):    6.01%
 Time  (mean ± σ):   1.365 s ± 35.318 ms  ┊ GC (mean ± σ):  5.52% ± 2.00%

  ▁▁      ▁   ▁         █▁       ▁▁ ▁                     ▁
  ██▁▁▁▁▁▁█▁▁▁█▁▁▁▁▁▁▁▁▁██▁▁▁▁▁▁▁██▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  1.32 s         Histogram: frequency by time        1.44 s <

 Memory estimate: 2.00 GiB, allocs estimate: 138279.
# MetidaCu
BenchmarkTools.Trial: 3 samples with 1 evaluation.
 Range (min … max):  7.391 s …   7.432 s  ┊ GC (min … max): 1.10% … 1.43%
 Time  (median):     7.426 s              ┊ GC (median):    1.30%
 Time  (mean ± σ):   7.416 s ± 21.787 ms  ┊ GC (mean ± σ):  1.28% ± 0.17%

  █                                               █       █
  █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁█ ▁
  7.39 s         Histogram: frequency by time        7.43 s <

 Memory estimate: 1.87 GiB, allocs estimate: 919345.


MODEL 5
# Metida
BenchmarkTools.Trial: 1 sample with 1 evaluation.
 Single result which took 216.945 s (0.04% GC) to evaluate,
 with a memory estimate of 3.91 GiB, over 7229 allocations.
# MetidaNLopt
BenchmarkTools.Trial: 2 samples with 1 evaluation.
 Range (min … max):  8.973 s …    9.139 s  ┊ GC (min … max): 3.06% … 2.77%
 Time  (median):     9.056 s               ┊ GC (median):    2.91%
 Time  (mean ± σ):   9.056 s ± 117.413 ms  ┊ GC (mean ± σ):  2.91% ± 0.20%

  █                                                        █
  █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  8.97 s         Histogram: frequency by time         9.14 s <

 Memory estimate: 7.86 GiB, allocs estimate: 57558.
# MetidaCu
BenchmarkTools.Trial: 2 samples with 1 evaluation.
 Range (min … max):  12.039 s …  12.084 s  ┊ GC (min … max): 1.85% … 1.78%
 Time  (median):     12.062 s              ┊ GC (median):    1.81%
 Time  (mean ± σ):   12.062 s ± 31.774 ms  ┊ GC (mean ± σ):  1.81% ± 0.05%

  █                                                        █
  █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  12 s           Histogram: frequency by time         12.1 s <

 Memory estimate: 8.31 GiB, allocs estimate: 365031.

=#
