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
random   = VarEffect(@covstr(formulation), CSH),
repeated = VarEffect(@covstr(formulation), VC),
subject  = :subject)

fit!(lmm; solver = :nlopt)
```
