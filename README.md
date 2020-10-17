# Metida

[![Project Status: Concept – Minimal or no implementation has been done yet, or the repository is only intended to be a limited example, demo, or proof-of-concept.](https://www.repostatus.org/badges/latest/concept.svg)](https://www.repostatus.org/#concept)

Metida.jl is a experimental Julia package for fitting mixed-effects models with flexible covariance structure. At this moment package is in early development stage.

*Alfa version*

Install:

```
import Pkg; Pkg.add("Metida")
```

Using:

```
using Metida, StatsBase, StatsModels, CSV, DataFrames
df = CSV.File(dirname(pathof(Metida))*"\\..\\test\\csv\\df0.csv") |> DataFrame
categorical!(df, :subject);
categorical!(df, :period);
categorical!(df, :sequence);
categorical!(df, :formulation);

lmm = LMM(@formula(var~sequence+period+formulation), df;
random = VarEffect(@covstr(formulation), CSH),
repeated = VarEffect(@covstr(formulation), VC),
subject = :subject)

fit!(lmm)
```

© 2020 Metida
