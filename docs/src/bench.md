# Benchmark

### System

* CPU: Ryzen 5950x
* RAM: 64Gb 3200

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
BenchmarkTools.Trial:
  memory estimate:  372.62 KiB
  allocs estimate:  3503
  --------------
  minimum time:     1.507 ms (0.00% GC)
  median time:      1.583 ms (0.00% GC)
  mean time:        1.652 ms (1.94% GC)
  maximum time:     12.631 ms (0.00% GC)
  --------------
  samples:          9053
  evals/sample:     1
```

### Metida

```
lmm = LMM(@formula(response ~1 + factor*time), rds;
random = VarEffect(@covstr(1 + time|subject&factor), CSH),
)
@benchmark fit!($lmm, hes = false) seconds = 15
```

* Metida v0.11.0

```
BenchmarkTools.Trial: 445 samples with 1 evaluation.
 Range (min … max):  20.060 ms … 157.976 ms  ┊ GC (min … max):  0.00% … 79.83%
 Time  (median):     26.957 ms               ┊ GC (median):     0.00%
 Time  (mean ± σ):   33.721 ms ±  26.626 ms  ┊ GC (mean ± σ):  19.89% ± 19.16%

  ▂▃██▄  
  █████▆▆▁▅▄▄▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▅▁▅▁▄▅▁▅█▄▁▅▁▅▁▁▄▅ ▆
  20.1 ms       Histogram: log(frequency) by time       152 ms <

 Memory estimate: 58.33 MiB, allocs estimate: 251271.
```

* MetidaNLopt v0.1.* (Metida 0.4)

```
@benchmark fit!($lmm, solver = :nlopt) seconds = 15
```

```
BenchmarkTools.Trial:
  memory estimate:  9.12 MiB
  allocs estimate:  53289
  --------------
  minimum time:     113.572 ms (0.00% GC)
  median time:      129.960 ms (0.00% GC)
  mean time:        131.570 ms (1.13% GC)
  maximum time:     167.145 ms (0.00% GC)
  --------------
  samples:          114
  evals/sample:     1
```

* MetidaCu v0.1.* (Metida 0.4)

```
@benchmark fit!($lmm, solver = :cuda) seconds = 15
```

```
BenchmarkTools.Trial:
  memory estimate:  26.44 MiB
  allocs estimate:  527567
  --------------
  minimum time:     3.782 s (0.00% GC)
  median time:      3.911 s (0.00% GC)
  mean time:        3.947 s (0.32% GC)
  maximum time:     4.183 s (1.20% GC)
  --------------
  samples:          4
  evals/sample:     1
```

### Cancer data:

File: [hdp.csv](https://stats.idre.ucla.edu/stat/data/hdp.csv) , 8525 observations.

#### Model 1: maximum 377 observation-per-subject (35 subjects)

```
lmm = Metida.LMM(@formula(tumorsize ~ 1 + CancerStage), hdp;
random = Metida.VarEffect(Metida.@covstr(1|HID), Metida.DIAG),
)
```

* Metida v0.10.1 Theads

```
julia> @benchmark  Metida.fit!(lmm, hes = false)
BenchmarkTools.Trial: 1 sample with 1 evaluation.
 Single result which took 10.214 s (1.64% GC) to evaluate,
 with a memory estimate of 3.64 GiB, over 103755 allocations.
```

* MetidaNLopt v0.3.2 (Metida v0.10.1)

```
julia> @benchmark  Metida.fit!(lmm; solver = :nlopt)
BenchmarkTools.Trial: 9 samples with 1 evaluation.
 Range (min … max):  447.502 ms … 680.028 ms  ┊ GC (min … max):  0.00% … 9.85%
 Time  (median):     575.979 ms               ┊ GC (median):    10.36%
 Time  (mean ± σ):   567.062 ms ±  81.027 ms  ┊ GC (mean ± σ):   8.53% ± 6.50%

  ▁          ▁    █                ▁       ▁      ▁    ▁      ▁
  █▁▁▁▁▁▁▁▁▁▁█▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁█▁▁▁▁▁▁█▁▁▁▁█▁▁▁▁▁▁█ ▁
  448 ms           Histogram: frequency by time          680 ms <

 Memory estimate: 920.57 MiB, allocs estimate: 51647.
```

* MetidaCu v0.2.0 (Metida 0.5.1)

```
julia> @benchmark  Metida.fit!(lmm; solver = :cuda)
BenchmarkTools.Trial:
  memory estimate:  1.02 GiB
  allocs estimate:  541438
  --------------
  minimum time:     6.935 s (1.07% GC)
  median time:      6.935 s (1.07% GC)
  mean time:        6.935 s (1.07% GC)
  maximum time:     6.935 s (1.07% GC)
  --------------
  samples:          1
  evals/sample:     1
```

#### Model 2: maximum 875 observation-per-subject (20 subjects)

```
lmm = Metida.LMM(@formula(tumorsize ~ 1 + CancerStage), hdp;
random = Metida.VarEffect(Metida.@covstr(1|Experience), Metida.SI),
)
```

* MetidaNLopt v0.2.0 (Metida 0.5.1)

```
julia> @benchmark  Metida.fit!(lmm; solver = :nlopt)
BenchmarkTools.Trial:
  memory estimate:  1.79 GiB
  allocs estimate:  17215
  --------------
  minimum time:     12.168 s (2.24% GC)
  median time:      12.168 s (2.24% GC)
  mean time:        12.168 s (2.24% GC)
  maximum time:     12.168 s (2.24% GC)
  --------------
  samples:          1
  evals/sample:     1
```

* MetidaCu v0.2.0 (Metida 0.5.1)

```
julia> @benchmark  Metida.fit!(lmm; solver = :cuda)
BenchmarkTools.Trial:
  memory estimate:  1.77 GiB
  allocs estimate:  274940
  --------------
  minimum time:     8.926 s (2.83% GC)
  median time:      8.926 s (2.83% GC)
  mean time:        8.926 s (2.83% GC)
  maximum time:     8.926 s (2.83% GC)
  --------------
  samples:          1
  evals/sample:     1
```

#### Model 3: maximum 1437 observation-per-subject (10 subjects)

```
lmm = Metida.LMM(@formula(tumorsize ~ 1 + CancerStage), hdp;
random = Metida.VarEffect(Metida.@covstr(1|ntumors), Metida.SI),
)
```

* MetidaNLopt v0.2.0 (Metida 0.5.1)

```
julia> @benchmark  Metida.fit!(lmm; solver = :nlopt)
BenchmarkTools.Trial:
  memory estimate:  5.28 GiB
  allocs estimate:  12951
  --------------
  minimum time:     32.109 s (2.24% GC)
  median time:      32.109 s (2.24% GC)
  mean time:        32.109 s (2.24% GC)
  maximum time:     32.109 s (2.24% GC)
  --------------
  samples:          1
  evals/sample:     1
```

* MetidaCu v0.2.0 (Metida 0.5.1)

```
julia> @benchmark  Metida.fit!(lmm; solver = :cuda)
BenchmarkTools.Trial:
  memory estimate:  4.36 GiB
  allocs estimate:  174131
  --------------
  minimum time:     22.690 s (2.89% GC)
  median time:      22.690 s (2.89% GC)
  mean time:        22.690 s (2.89% GC)
  maximum time:     22.690 s (2.89% GC)
  --------------
  samples:          1
  evals/sample:     1
```

#### Model 4: maximum 3409 observation-per-subject (4 subjects)

* MetidaCu v0.2.0 (Metida 0.5.1)

```
lmm = Metida.LMM(@formula(tumorsize ~ 1 + CancerStage), hdp;
random = Metida.VarEffect(Metida.@covstr(1|CancerStage), Metida.SI),
)
```

```
julia> @benchmark  Metida.fit!(lmm; solver = :cuda)
BenchmarkTools.Trial:
  memory estimate:  5.69 GiB
  allocs estimate:  43924
  --------------
  minimum time:     28.227 s (2.73% GC)
  median time:      28.227 s (2.73% GC)
  mean time:        28.227 s (2.73% GC)
  maximum time:     28.227 s (2.73% GC)
  --------------
  samples:          1
  evals/sample:     1
```

### Conclusion

MixedModels.jl faster than Metida.jl in similar cases, but Metida.jl can be used with different covariance structures for random and repeated effects. MetidaNLopt have better performance but not estimate Hessian matrix of REML. MetidaCu have advantage only for big observation-pes-subject number.
