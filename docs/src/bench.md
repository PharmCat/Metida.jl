# Benchmark

* Data

```
using Metida, CSV, DataFrames, MixedModels;

rds = CSV.File(joinpath(dirname(pathof(Metida)), "..", "test", "csv",  "1fptime.csv"); types = [String, String, Float64, Float64]) |> DataFrame
```

### MixedModels

```
fm = @formula(response ~ 1 + factor*time + (1 + time|subject&factor))
@benchmark mm = fit($MixedModel, $fm, $rds, REML=true) seconds = 15
```

```
BenchmarkTools.Trial:
  memory estimate:  375.98 KiB
  allocs estimate:  3544
  --------------
  minimum time:     4.797 ms (0.00% GC)
  median time:      5.232 ms (0.00% GC)
  mean time:        5.469 ms (1.43% GC)
  maximum time:     32.592 ms (78.90% GC)
  --------------
  samples:          2741
  evals/sample:     1
```

### Metida

```
lmm = LMM(@formula(response ~1 + factor*time), rds;
random = VarEffect(@covstr(1 + time|subject&factor), CSH),
)
@benchmark fit!($lmm, hes = false) seconds = 15
```

* Metida v0.4.0

```
BenchmarkTools.Trial:
  memory estimate:  74.98 MiB
  allocs estimate:  112761
  --------------
  minimum time:     198.118 ms (0.00% GC)
  median time:      212.742 ms (3.56% GC)
  mean time:        212.262 ms (2.83% GC)
  maximum time:     225.154 ms (4.37% GC)
  --------------
  samples:          71
  evals/sample:     1
```

* Metida v0.5.0

```
BenchmarkTools.Trial:
  memory estimate:  74.93 MiB
  allocs estimate:  111421
  --------------
  minimum time:     196.160 ms (0.00% GC)
  median time:      218.341 ms (0.00% GC)
  mean time:        225.815 ms (3.55% GC)
  maximum time:     305.821 ms (19.83% GC)
  --------------
  samples:          67
  evals/sample:     1
```

* Metida v0.7.0

```
BenchmarkTools.Trial:
  memory estimate:  46.25 MiB
  allocs estimate:  118271
  --------------
  minimum time:     92.129 ms (0.00% GC)
  median time:      99.525 ms (0.00% GC)
  mean time:        105.830 ms (4.89% GC)
  maximum time:     184.984 ms (0.00% GC)
  --------------
  samples:          142
  evals/sample:     1
```

* MetidaNLopt v0.1.*

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

* MetidaCu v0.1.*

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

* MetidaNLopt v0.2.0 (Metida 0.5.1)

```
julia> @benchmark  Metida.fit!(lmm; solver = :nlopt)
BenchmarkTools.Trial:
  memory estimate:  1.03 GiB
  allocs estimate:  32925
  --------------
  minimum time:     7.882 s (0.75% GC)
  median time:      7.882 s (0.75% GC)
  mean time:        7.882 s (0.75% GC)
  maximum time:     7.882 s (0.75% GC)
  --------------
  samples:          1
  evals/sample:     1
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
