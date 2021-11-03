# Benchmark

### System

* CPU: Ryzen 5950x
* RAM: 64Gb 3200
* GTX 1070Ti

### Data

```
using Metida, CSV, DataFrames, MixedModels, BenchmarkTools;

rds = CSV.File(joinpath(dirname(pathof(Metida)), "..", "test", "csv",  "1fptime.csv"); types = [String, String, Float64, Float64]) |> DataFrame
```

### MixedModels

```
fm = @formula(response ~ 1 + factor*time + (1 + time|subject&factor))
@benchmark mm = fit($MixedModel, $fm, $rds, REML=true) seconds = 15
```

```
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):  1.140 ms …  10.909 ms  ┊ GC (min … max): 0.00% … 86.12%
 Time  (median):     1.175 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   1.215 ms ± 563.839 μs  ┊ GC (mean ± σ):  2.78% ±  5.31%

          ▄██▆▃▁          
  ▁▁▁▂▂▃▅▇██████▇▅▄▃▃▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ ▂
  1.14 ms         Histogram: frequency by time        1.33 ms <

 Memory estimate: 409.52 KiB, allocs estimate: 6130.
```

### Metida

```
lmm = LMM(@formula(response ~1 + factor*time), rds;
random = VarEffect(@covstr(1 + time|subject&factor), CSH),
)
@benchmark fit!($lmm, hes = false) seconds = 15
```

* Metida v0.12.0

```
BenchmarkTools.Trial: 1316 samples with 1 evaluation.
 Range (min … max):   5.394 ms … 186.301 ms  ┊ GC (min … max):  0.00% … 95.48%
 Time  (median):      7.648 ms               ┊ GC (median):     0.00%
 Time  (mean ± σ):   11.391 ms ±  19.135 ms  ┊ GC (mean ± σ):  32.70% ± 17.73%

  ██▆
  ███▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▄▁▆▅▄▄▅▄▅▄▄▆▄▆▄▄▄ █
  5.39 ms       Histogram: log(frequency) by time       112 ms <

 Memory estimate: 22.63 MiB, allocs estimate: 37224.
```

* MetidaNLopt v0.4.0 (Metida 0.12.0)

```
@benchmark fit!($lmm, solver = :nlopt, hes = false, f_tol=1e-8, x_tol=1e-8) seconds = 15
```

```
BenchmarkTools.Trial: 274 samples with 1 evaluation.
 Range (min … max):  47.312 ms … 153.284 ms  ┊ GC (min … max):  0.00% … 67.58%
 Time  (median):     49.064 ms               ┊ GC (median):     0.00%
 Time  (mean ± σ):   54.854 ms ±  19.559 ms  ┊ GC (mean ± σ):  10.55% ± 15.98%

  ▅█    
  ███▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▆▅▁▁▄▁▁▄▁▄▁▁▄▄▅▁▁▁▁▁▁▁▁▁▁▁▄▁▄▄▁▅▄▁▅▄▄ ▅
  47.3 ms       Histogram: log(frequency) by time       135 ms <

 Memory estimate: 35.45 MiB, allocs estimate: 301141.
```

* MetidaCu v0.4.0 (Metida 0.4)

```
@benchmark fit!($lmm, solver = :cuda, hes = false, f_tol=1e-8, x_tol=1e-8) seconds = 15
```

```
BenchmarkTools.Trial: 42 samples with 1 evaluation.
 Range (min … max):  347.642 ms … 461.104 ms  ┊ GC (min … max): 0.00% … 4.12%
 Time  (median):     350.603 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   358.874 ms ±  23.939 ms  ┊ GC (mean ± σ):  0.27% ± 0.98%

  ▁█     
  ███▁▃▁▁▁▁▁▁▁▁▅▄▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▃▁▁▃ ▁
  348 ms           Histogram: frequency by time          461 ms <

 Memory estimate: 6.86 MiB, allocs estimate: 115020.
```

### Cancer data:

File: [hdp.csv](https://stats.idre.ucla.edu/stat/data/hdp.csv) , 8525 observations.

#### Model 1: maximum 377 observation-per-subject (35 subjects)

```
lmm = Metida.LMM(@formula(tumorsize ~ 1 + CancerStage), hdp;
random = Metida.VarEffect(Metida.@covstr(1|HID), Metida.DIAG),
)
```

* Metida v0.12.0

```
@benchmark  Metida.fit!(lmm, hes = false)
```

```
BenchmarkTools.Trial: 1 sample with 1 evaluation.
 Single result which took 6.519 s (1.38% GC) to evaluate,
 with a memory estimate of 2.33 GiB, over 41654 allocations.
```

* MetidaNLopt v0.4.0 (Metida v0.12.0)

```
@benchmark fit!($lmm, solver = :nlopt, hes = false, f_tol=1e-8, x_tol=1e-8) seconds = 15
```

```
BenchmarkTools.Trial: 25 samples with 1 evaluation.
 Range (min … max):  555.136 ms … 700.605 ms  ┊ GC (min … max): 0.00% … 16.09%
 Time  (median):     605.713 ms               ┊ GC (median):    7.98%
 Time  (mean ± σ):   608.768 ms ±  27.220 ms  ┊ GC (mean ± σ):  7.62% ±  3.82%

                    █    ▂
  ▅▁▁▅▁▁▁▁▁▅▁▁▁▁▁▅▁▅█▅▅▅▅█▅▁▁▅▅▁▅█▁▁▅▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▅ ▁
  555 ms           Histogram: frequency by time          701 ms <

 Memory estimate: 921.54 MiB, allocs estimate: 62203.
```

* MetidaCu v0.4.0 (Metida 0.12.0)

```
@benchmark fit!($lmm, solver = :cuda, hes = false, f_tol=1e-8, x_tol=1e-8) seconds = 15
```

```
BenchmarkTools.Trial: 5 samples with 1 evaluation.
 Range (min … max):  3.482 s …   3.650 s  ┊ GC (min … max): 0.00% … 2.73%
 Time  (median):     3.496 s              ┊ GC (median):    0.00%
 Time  (mean ± σ):   3.547 s ± 77.924 ms  ┊ GC (mean ± σ):  1.07% ± 1.43%

  ▁   █                                       ▁           ▁
  █▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁█ ▁
  3.48 s         Histogram: frequency by time        3.65 s <

 Memory estimate: 913.96 MiB, allocs estimate: 410438.
```

#### Model 2: maximum 875 observation-per-subject (20 subjects)

```
lmm = Metida.LMM(@formula(tumorsize ~ 1 + CancerStage), hdp;
random = Metida.VarEffect(Metida.@covstr(1|Experience), Metida.SI),
)
```

* MetidaNLopt v0.2.0 (Metida 0.5.1)

```
@benchmark fit!($lmm, solver = :nlopt, hes = false, f_tol=1e-8, x_tol=1e-8) seconds = 1
```

```
BenchmarkTools.Trial: 9 samples with 1 evaluation.
 Range (min … max):  1.651 s …    1.952 s  ┊ GC (min … max): 3.10% … 4.27%
 Time  (median):     1.797 s               ┊ GC (median):    4.15%
 Time  (mean ± σ):   1.815 s ± 101.277 ms  ┊ GC (mean ± σ):  3.83% ± 0.82%

  ▁               ▁  ▁ ▁      ▁            ▁ ▁             █
  █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁█▁█▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁█▁█▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  1.65 s         Histogram: frequency by time         1.95 s <

 Memory estimate: 2.47 GiB, allocs estimate: 57729.
```

* MetidaCu v0.2.0 (Metida 0.5.1)

```
@benchmark fit!($lmm, solver = :cuda, hes = false, f_tol=1e-8, x_tol=1e-8) seconds = 15
```

```
BenchmarkTools.Trial: 3 samples with 1 evaluation.
 Range (min … max):  5.137 s …   5.216 s  ┊ GC (min … max): 1.03% … 2.19%
 Time  (median):     5.166 s              ┊ GC (median):    1.43%
 Time  (mean ± σ):   5.173 s ± 39.699 ms  ┊ GC (mean ± σ):  1.55% ± 0.59%

  █                   █                                   █
  █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  5.14 s         Histogram: frequency by time        5.22 s <

 Memory estimate: 2.46 GiB, allocs estimate: 372716.
```

#### Model 3: maximum 1437 observation-per-subject (10 subjects)

```
lmm = Metida.LMM(@formula(tumorsize ~ 1 + CancerStage), hdp;
random = Metida.VarEffect(Metida.@covstr(1|ntumors), Metida.SI),
)
```

* MetidaNLopt v0.4.0 (Metida 0.12.0)

```
@benchmark fit!($lmm, solver = :nlopt, hes = false, f_tol=1e-8, x_tol=1e-8) seconds = 15
```

```
BenchmarkTools.Trial: 4 samples with 1 evaluation.
 Range (min … max):  4.305 s …   4.476 s  ┊ GC (min … max): 2.49% … 3.52%
 Time  (median):     4.372 s              ┊ GC (median):    3.23%
 Time  (mean ± σ):   4.381 s ± 80.689 ms  ┊ GC (mean ± σ):  3.12% ± 0.47%

  █     █                               █                 █
  █▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  4.3 s          Histogram: frequency by time        4.48 s <

 Memory estimate: 3.83 GiB, allocs estimate: 28068.
```

* MetidaCu v0.4.0 (Metida 0.12.0)

```
@benchmark fit!($lmm, solver = :cuda, hes = false, f_tol=1e-8, x_tol=1e-8) seconds = 15
```

```
BenchmarkTools.Trial: 4 samples with 1 evaluation.
 Range (min … max):  4.928 s …   4.970 s  ┊ GC (min … max): 1.83% … 1.78%
 Time  (median):     4.957 s              ┊ GC (median):    1.85%
 Time  (mean ± σ):   4.953 s ± 18.996 ms  ┊ GC (mean ± σ):  1.90% ± 0.15%

  █                          █                       █    █
  █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁█ ▁
  4.93 s         Histogram: frequency by time        4.97 s <

 Memory estimate: 3.39 GiB, allocs estimate: 149182.
```

#### Model 4: maximum 3409 observation-per-subject (4 subjects)

* MetidaCu v0.4.0 (Metida 0.12.0)

```
lmm = Metida.LMM(@formula(tumorsize ~ 1 + CancerStage), hdp;
random = Metida.VarEffect(Metida.@covstr(1|CancerStage), Metida.SI),
)
```

```
@benchmark fit!($lmm, solver = :cuda, hes = false, f_tol=1e-8, x_tol=1e-8) seconds = 15
```

```
julia> @benchmark fit!($lmm, solver = :cuda, hes = false, f_tol=1e-8, x_tol=1e-8) seconds = 15
BenchmarkTools.Trial: 3 samples with 1 evaluation.
 Range (min … max):  7.343 s …   7.372 s  ┊ GC (min … max): 1.62% … 1.48%
 Time  (median):     7.346 s              ┊ GC (median):    1.49%
 Time  (mean ± σ):   7.354 s ± 15.657 ms  ┊ GC (mean ± σ):  1.50% ± 0.11%

  █    █                                                  █
  █▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  7.34 s         Histogram: frequency by time        7.37 s <

 Memory estimate: 5.04 GiB, allocs estimate: 46549.
```

### Conclusion

MixedModels.jl faster than Metida.jl in similar cases, but Metida.jl can be used with different covariance structures for random and repeated effects. MetidaNLopt have better performance but not estimate Hessian matrix of REML. MetidaCu have advantage only for big observation-pes-subject number.
