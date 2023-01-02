### NLopt

Optimization with NLopt.jl.

Install:

```
using Pkg
Pkg.add("MetidaNLopt")
```

Using:

```
using Metida, MetidaNLopt, StatsBase, StatsModels, CSV, DataFrames
df = CSV.File(dirname(pathof(Metida))*"\\..\\test\\csv\\df0.csv") |> DataFrame
lmm = LMM(@formula(var ~ sequence + period + formulation), df;
random   = VarEffect(@covstr(formulation|subject), CSH),
repeated = VarEffect(@covstr(formulation|subject), VC))
fit!(lmm; solver = :nlopt)
```


NLopt is a free/open-source library for nonlinear optimization, providing a common interface for a number of different free optimization routines available online as well as original implementations of various other algorithms.


Optimization with NLopt.jl using gradient-free algirithms is less stable, that why two-step optimization schema used. Results can be slightly different for differens OS and Julia versions. Always look into logs. 