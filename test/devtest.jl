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
lmm = Metida.LMM(@formula(response ~1 + factor*time), ftdf;
random = Metida.VarEffect(Metida.@covstr(1 + time|subject&factor), Metida.CSH),
)
b[1] = @benchmark Metida.fit!($lmm, hes = false; maxthreads = 16) seconds = 15
# MODEL 2
lmm = Metida.LMM(@formula(response ~1 + factor*time), ftdf;
random = Metida.VarEffect(Metida.@covstr(1 + time|factor), Metida.ARH),
)
b[2]  = @benchmark Metida.fit!($lmm, hes = false; maxthreads = 16) seconds = 15
# MODEL 3
lmm = Metida.LMM(@formula(response ~1 + factor*time), ftdf;
random = Metida.VarEffect(Metida.@covstr(1 + time|subject&factor), Metida.CSH),
repeated = Metida.VarEffect(Metida.@covstr(1|subject), Metida.AR),
)
b[3]  = @benchmark Metida.fit!($lmm, hes = false; maxthreads = 16) seconds = 15
# MODEL 4
lmm = Metida.LMM(@formula(tumorsize ~ 1 + CancerStage), hdp;
random = Metida.VarEffect(Metida.@covstr(1|HID), Metida.DIAG),
)
b[4]  = @benchmark Metida.fit!($lmm, hes = false; maxthreads = 16) seconds = 15

for i = 1:4
  display(b[i])
  push!(results, (now(), "Model $i", minimum(b[i]).time, minimum(b[i]).memory, minimum(b[i]).allocs))
end

#=
BenchmarkTools.Trial: 411 samples with 1 evaluation.
 Range (min … max):  11.681 ms …    1.511 s  ┊ GC (min … max):  0.00% … 99.24%
 Time  (median):     14.456 ms               ┊ GC (median):     0.00%
 Time  (mean ± σ):   36.240 ms ± 170.372 ms  ┊ GC (mean ± σ):  57.08% ± 11.90%

  █ 
  █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▄▁▁▁▁▁▁▁▁▁▄ ▆
  11.7 ms       Histogram: log(frequency) by time       1.42 s <

 Memory estimate: 19.38 MiB, allocs estimate: 31265.
BenchmarkTools.Trial: 23 samples with 1 evaluation.
 Range (min … max):  657.256 ms … 719.396 ms  ┊ GC (min … max): 0.00% … 4.31%
 Time  (median):     670.301 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   677.402 ms ±  16.735 ms  ┊ GC (mean ± σ):  1.46% ± 2.20%

        █ ▁            
  ▆▆▆▁▁▆█▆█▁▁▁▆▁▁▁▁▁▁▁▁▁▁▆▆▆▁▁▁▆▆▆▁▁▁▁▆▁▆▆▁▆▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▆ ▁
  657 ms           Histogram: frequency by time          719 ms <

 Memory estimate: 132.83 MiB, allocs estimate: 5086.
BenchmarkTools.Trial: 297 samples with 1 evaluation.
 Range (min … max):  17.368 ms …    1.552 s  ┊ GC (min … max):  0.00% … 98.55%
 Time  (median):     20.409 ms               ┊ GC (median):     0.00%
 Time  (mean ± σ):   54.268 ms ± 197.857 ms  ┊ GC (mean ± σ):  60.26% ± 15.96%

  █
  █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▄▁▁▅▅ ▅
  17.4 ms       Histogram: log(frequency) by time       1.19 s <

 Memory estimate: 42.28 MiB, allocs estimate: 23492.
BenchmarkTools.Trial: 4 samples with 1 evaluation.
 Range (min … max):  4.696 s …    4.993 s  ┊ GC (min … max): 2.54% … 8.32%
 Time  (median):     4.727 s               ┊ GC (median):    2.74%
 Time  (mean ± σ):   4.786 s ± 138.981 ms  ┊ GC (mean ± σ):  4.11% ± 2.86%

  █    ██                                                  █
  █▁▁▁▁██▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  4.7 s          Histogram: frequency by time         4.99 s <

 Memory estimate: 2.56 GiB, allocs estimate: 40772.
=#