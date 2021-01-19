## CUDA

Use CuBLAS & CuSOLVER for REML calculation. Optimization with NLopt.jl.

Install:

```
using Pkg
Pkg.add("MetidaCu")
```

Using:

```
using Metida, MetidaCu, StatsBase, StatsModels, CSV, DataFrames
df = CSV.File(dirname(pathof(Metida))*"\\..\\test\\csv\\df0.csv") |> DataFrame
lmm = LMM(@formula(var ~ sequence + period + formulation), df;
random   = VarEffect(@covstr(formulation), CSH),
repeated = VarEffect(@covstr(formulation), VC),
subject  = :subject)

fit!(lmm; solver = :cuda)
```
