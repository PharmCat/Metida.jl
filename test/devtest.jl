#using NLopt
using Metida

using DataFrames, CSV, StatsModels, LinearAlgebra, CategoricalArrays, Dates

using BenchmarkTools
path    = dirname(@__FILE__)
cd(path)
df0         = CSV.File(path*"/csv/df0.csv"; types = [String, String, String, String, Float64, Float64, Float64]) |> DataFrame
df1         = CSV.File(path*"/csv/df1.csv"; types = [String, String, String, String, Float64, Float64]) |> DataFrame
ftdf         = CSV.File(path*"/csv/1fptime.csv"; types = [String, String, Float64, Float64]) |> DataFrame
ftdf2        = CSV.File(path*"/csv/1freparma.csv"; types = [String, String, Float64, Float64]) |> DataFrame
ftdf3        = CSV.File(path*"/csv/ftdf3.csv"; types =
[String,  Float64, Float64, String, String, String, String, String, Float64]) |> DataFrame
hdp         = CSV.File("hdp.csv") |> DataFrame
transform!(hdp, :DID => categorical);
transform!(hdp, :HID=> categorical);
transform!(hdp, :Sex=> categorical);
transform!(hdp, :School=> categorical);
transform!(hdp, :pain=> categorical);
pkgversion(m::Module) = Pkg.TOML.parsefile(joinpath(dirname(string(first(methods(m.eval)).file)), "..", "Project.toml"))["version"]


results = DataFrame(datetime =[], model = [], mintime =[], memory = [], allocs = [])
b = Vector{Any}(undef, 4)
################################################################################
# Metida
################################################################################
# MODEL 1
println("MODEL 1")
lmm = Metida.LMM(@formula(response ~1 + factor*time), ftdf;
random = Metida.VarEffect(Metida.@covstr(1 + time|subject&factor), Metida.CSH),
)
b[1] = @benchmark Metida.fit!($lmm, hes = false; maxthreads = 16) seconds = 15
# MODEL 2
println("MODEL 2")
lmm = Metida.LMM(@formula(response ~1 + factor*time), ftdf;
random = Metida.VarEffect(Metida.@covstr(1 + time|factor), Metida.ARH),
)
b[2]  = @benchmark Metida.fit!($lmm, hes = false; maxthreads = 16) seconds = 15
# MODEL 3
println("MODEL 3")
lmm = Metida.LMM(@formula(response ~1 + factor*time), ftdf;
random = Metida.VarEffect(Metida.@covstr(1 + time|subject&factor), Metida.CSH),
repeated = Metida.VarEffect(Metida.@covstr(1|subject), Metida.AR),
)
b[3]  = @benchmark Metida.fit!($lmm, hes = false; maxthreads = 16) seconds = 15
# MODEL 4
println("MODEL 4")
lmm = Metida.LMM(@formula(tumorsize ~ 1 + CancerStage), hdp;
random = Metida.VarEffect(Metida.@covstr(1|HID), Metida.DIAG),
)
b[4]  = @benchmark Metida.fit!($lmm, hes = false; maxthreads = 16) seconds = 15

for i = 1:4
  println("MODEL $i")
  display(b[i])
  push!(results, (now(), "Model $i", minimum(b[i]).time, minimum(b[i]).memory, minimum(b[i]).allocs))
end

#=
MODEL 1
BenchmarkTools.Trial: 675 samples with 1 evaluation.
 Range (min … max):   8.194 ms … 798.405 ms  ┊ GC (min … max):  0.00% … 98.64%
 Time  (median):     10.882 ms               ┊ GC (median):     0.00%
 Time  (mean ± σ):   22.823 ms ±  81.034 ms  ┊ GC (mean ± σ):  52.70% ± 14.48%

  █  
  █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▃▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▄▄▁▄▃ ▆
  8.19 ms       Histogram: log(frequency) by time       533 ms <

 Memory estimate: 19.38 MiB, allocs estimate: 31265.
MODEL 2
BenchmarkTools.Trial: 23 samples with 1 evaluation.
 Range (min … max):  657.833 ms … 704.073 ms  ┊ GC (min … max): 0.00% … 3.23%
 Time  (median):     675.148 ms               ┊ GC (median):    0.00%        
 Time  (mean ± σ):   677.822 ms ±  12.920 ms  ┊ GC (mean ± σ):  1.41% ± 1.62%

  ▁     ▁▁ ▁ ██  ▁ ▁   ▁▁   ▁  ▁  ▁   ▁   █    ▁ █   ▁        ▁  
  █▁▁▁▁▁██▁█▁██▁▁█▁█▁▁▁██▁▁▁█▁▁█▁▁█▁▁▁█▁▁▁█▁▁▁▁█▁█▁▁▁█▁▁▁▁▁▁▁▁█ ▁
  658 ms           Histogram: frequency by time          704 ms <

 Memory estimate: 132.83 MiB, allocs estimate: 5085.
MODEL 3
BenchmarkTools.Trial: 440 samples with 1 evaluation.
 Range (min … max):  13.693 ms … 644.360 ms  ┊ GC (min … max):  0.00% … 97.15%
 Time  (median):     17.967 ms               ┊ GC (median):     0.00%
 Time  (mean ± σ):   34.118 ms ±  86.080 ms  ┊ GC (mean ± σ):  47.99% ± 18.02%

  █  
  █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▄▄▄▄▁▅▄▄▄▄▁▁▁▁▁▁▄ ▆
  13.7 ms       Histogram: log(frequency) by time       510 ms <

 Memory estimate: 42.28 MiB, allocs estimate: 23491.
MODEL 4
BenchmarkTools.Trial: 4 samples with 1 evaluation.
 Range (min … max):  4.635 s …    4.880 s  ┊ GC (min … max): 1.77% … 4.69%
 Time  (median):     4.791 s               ┊ GC (median):    3.57%
 Time  (mean ± σ):   4.774 s ± 102.530 ms  ┊ GC (mean ± σ):  3.42% ± 1.34%

  █                                █      █                █
  █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  4.64 s         Histogram: frequency by time         4.88 s <

 Memory estimate: 2.56 GiB, allocs estimate: 40772.
=#


#= V 0.17.3
MODEL 1
BenchmarkTools.Trial: 2034 samples with 1 evaluation per sample.
 Range (min … max):  3.753 ms … 67.537 ms  ┊ GC (min … max):  0.00% … 89.18%
 Time  (median):     6.294 ms              ┊ GC (median):     0.00%
 Time  (mean ± σ):   7.342 ms ±  3.485 ms  ┊ GC (mean ± σ):  18.27% ± 21.03%

           ▃▃▄█▇▃                                             
  ▂▄▆▅▆▇▇█████████▅▃▃▃▃▂▂▂▂▁▁▁▁▁▁▁▂▁▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▃▄▄▄▄▄▃▃▂ ▃
  3.75 ms        Histogram: frequency by time        15.7 ms <

 Memory estimate: 7.90 MiB, allocs estimate: 30857.
MODEL 2
BenchmarkTools.Trial: 35 samples with 1 evaluation per sample.
 Range (min … max):  407.308 ms … 444.749 ms  ┊ GC (min … max): 0.00% … 4.51%
 Time  (median):     431.183 ms               ┊ GC (median):    4.70%
 Time  (mean ± σ):   430.217 ms ±   6.817 ms  ┊ GC (mean ± σ):  4.77% ± 1.33%

                          ▄    ▁   ▁   ▁▁█▁   ▄     ▁            
  ▆▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▆▁▁█▁▁▆▆█▁▁▆█▁▆▆████▁▆▆█▆▁▁▁▁█▆▁▁▆▁▁▁▁▁▆ ▁
  407 ms           Histogram: frequency by time          445 ms <

 Memory estimate: 130.96 MiB, allocs estimate: 6630.
MODEL 3
BenchmarkTools.Trial: 896 samples with 1 evaluation per sample.
 Range (min … max):   9.615 ms … 74.749 ms  ┊ GC (min … max):  0.00% … 13.00%
 Time  (median):     17.565 ms              ┊ GC (median):    31.07%
 Time  (mean ± σ):   16.735 ms ±  4.654 ms  ┊ GC (mean ± σ):  22.62% ± 17.46%

             ▃▄▇█▃                    ▂ ▁▅ ▁▁  ▁▁              
  ▂▂▁▃▃▄▄██▇███████▅▅▄▄▄▃▃▂▃▃▃▃▄▄▄▅▅▆▆█▇██████▇████▇▅▃▃▄▂▃▃▂▃ ▄
  9.62 ms         Histogram: frequency by time        23.8 ms <

 Memory estimate: 27.34 MiB, allocs estimate: 30315.
MODEL 4
BenchmarkTools.Trial: 4 samples with 1 evaluation per sample.
 Range (min … max):  4.239 s …   4.324 s  ┊ GC (min … max): 5.14% … 5.74%
 Time  (median):     4.275 s              ┊ GC (median):    5.73%
 Time  (mean ± σ):   4.278 s ± 34.902 ms  ┊ GC (mean ± σ):  5.60% ± 0.33%

  █                      █ █                              █  
  █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  4.24 s         Histogram: frequency by time        4.32 s <

 Memory estimate: 2.63 GiB, allocs estimate: 42704.
=#