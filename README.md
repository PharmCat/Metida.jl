# Metida

This program comes with absolutely no warranty. No liability is accepted for any loss and risk to public health resulting from use of this software.

| Version | pkgeval | Cover | Build | Docs |
|--------|-------|-------|-------|------|
|[![version](https://juliahub.com/docs/Metida/version.svg)](https://juliahub.com/ui/Packages/Metida/asHVr)|[![pkgeval](https://juliahub.com/docs/Metida/pkgeval.svg)](https://juliahub.com/ui/Packages/Metida/asHVr)|[![codecov](https://codecov.io/gh/PharmCat/Metida.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/PharmCat/Metida.jl)|![Tier 1](https://github.com/PharmCat/Metida.jl/workflows/Tier%201/badge.svg) | [![Latest docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://pharmcat.github.io/Metida.jl/dev/) [![Stable docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://pharmcat.github.io/Metida.jl/stable/)|

Metida.jl is Julia package for fitting mixed-effects models with flexible covariance structure.

Install:

```
import Pkg; Pkg.add("Metida")
```

Using:

```
using Metida, CSV, DataFrames, CategoricalArrays
df = CSV.File(joinpath(dirname(pathof(Metida)),"..","test","csv","df0.csv")) |> DataFrame
transform!(df, :subject => categorical, renamecols=false)
transform!(df, :period => categorical, renamecols=false)
transform!(df, :sequence => categorical, renamecols=false)
transform!(df, :formulation => categorical, renamecols=false)

lmm = LMM(@formula(var~sequence+period+formulation), df;
random = VarEffect(@covstr(formulation|subject), CSH),
repeated = VarEffect(@covstr(formulation|subject), DIAG),
)

fit!(lmm)

# Or you can use macro @lmmformula

lmm = LMM(@lmmformula(var~sequence+period+formulation,
    random = formulation|subject:CSH,
    repeated = formulation|subject:DIAG),
    df0)
fit!(lmm)
```

Also you can use this package with [MatidaNLopt.jl](https://github.com/PharmCat/MetidaNLopt.jl) and [MetidaCu.jl](https://github.com/PharmCat/MetidaCu.jl).

See also [MixedModels.jl](https://github.com/JuliaStats/MixedModels.jl): powerful package for mixed models.

Copyright © 2020 Metida Author: Vladimir Arnautov <mail@pharmcat.net>
