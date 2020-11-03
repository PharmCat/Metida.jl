# Metida

```@meta
CurrentModule = Metida
```

## Mixed Models

Metida.jl is a Julia package for fitting mixed-effects models with flexible covariance structure. At this moment package is in early development stage.

Main goal to make reproducible output corresponding to SAS/SPSS.

Now implemented covariance structures:

  * Scaled Identity (SI)
  * Variance Components / Diagonal (VC)
  * Autoregressive (AR)
  * Heterogeneous Autoregressive (ARH)
  * Compound Symmetry (CS)
  * Heterogeneous Compound Symmetry (CSH)

Usage:

`LMM(model, data; subject = nothing,  random = nothing, repeated = nothing)`

where

`model` is a fixed-effect model (`@formula`), example: `@formula(var ~ sequence + period + formulation)`

`random` vector of random effects or single random effect. Effect can be declared like this: `VarEffect(@covstr(formulation), CSH)`. `@covstr` is a effect model: `@covstr(formulation)`. `CSH` is a  CovarianceType structure. Premade constants: SI, VC, AR, ARH, CSH.

`repeated` is a repeated effect (only single).

`subject` is a block-diagonal factor.

```@contents
Pages = [
        "examples.md",
        "details.md",
        "api.md"]
Depth = 3
```

See also:

https://github.com/JuliaStats/MixedModels.jl
