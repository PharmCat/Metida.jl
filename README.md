# Metida

This program comes with absolutely no warranty. No liability is accepted for any loss and risk to public health resulting from use of this software.

| Status | Cover | Build | Docs |
|--------|-------|-------|------|
|[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)|[![codecov](https://codecov.io/gh/PharmCat/Metida.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/PharmCat/Metida.jl)|![Tier 1](https://github.com/PharmCat/Metida.jl/workflows/Tier%201/badge.svg) | [![Latest docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://pharmcat.github.io/Metida.jl/dev/)|

Metida.jl is a experimental Julia package for fitting mixed-effects models with flexible covariance structure. At this moment package is in development stage.


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
repeated = VarEffect(@covstr(formulation), DIAG),
subject = :subject)

fit!(lmm)
```

Also you can use this package with [MatidaNLopt.jl](https://github.com/PharmCat/MetidaNLopt.jl) and [MetidaCu.jl](https://github.com/PharmCat/MetidaCu.jl).

See also [MixedModels.jl](https://github.com/JuliaStats/MixedModels.jl): powerful package for mixed models. 

Copyright © 2020 Metida Author: Vladimir Arnautov <mail@pharmcat.net>
