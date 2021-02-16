### CUDA

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
random   = VarEffect(@covstr(formulation|subject), CSH),
repeated = VarEffect(@covstr(formulation|subject), VC))
fit!(lmm; solver = :cuda)
```
