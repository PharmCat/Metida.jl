# Metida

Experimental package for variance-component calculation.

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
