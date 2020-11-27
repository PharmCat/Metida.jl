# Metida

[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)

[![codecov](https://codecov.io/gh/PharmCat/Metida.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/PharmCat/Metida.jl)

[![Build Status](https://travis-ci.com/PharmCat/Metida.jl.svg?branch=master)](https://travis-ci.com/PharmCat/Metida.jl)

[![Latest docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://pharmcat.github.io/Metida.jl/dev/)

Metida.jl is a experimental Julia package for fitting mixed-effects models with flexible covariance structure. At this moment package is in early development stage.


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
