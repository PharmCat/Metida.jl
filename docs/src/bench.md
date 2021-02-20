# Benchmark

* Data

```
using Metida, CSV, DataFrames, MixedModels;

rds = CSV.File(joinpath(dirname(pathof(Metida)), "..", "test", "csv",  "1fptime.csv"); types = [String, String, Float64, Float64]) |> DataFrame
```

* MixedModels

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

* Metida v0.4.0

```
lmm = LMM(@formula(response ~1 + factor*time), rds;
random = VarEffect(@covstr(1 + time|subject&factor), CSH),
)
@benchmark fit!($lmm, hcalck = false) seconds = 15
```

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

* MetidaNLopt v0.4.0

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

* MetidaCu v0.4.0

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

* Metida v0.5.0

```
lmm = LMM(@formula(response ~1 + factor*time), rds;
random = VarEffect(@covstr(1 + time|subject&factor), CSH),
)
@benchmark fit!($lmm, hes = false) seconds = 15
```

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

### Conclusion

MixedModels.jl faster than Metida.jl in similar cases, but Metida.jl can be used with different covariance structures for random and repeated effects. MetidaNLopt have better performance but not estimate Hessian matrix of REML. MetidaCu have advantage only for big observation-pes-subject number.
