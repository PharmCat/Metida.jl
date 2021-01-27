# Metida

```@meta
CurrentModule = Metida
```

## Mixed Models

Metida.jl is a Julia package for fitting mixed-effects models with flexible covariance structure.

Implemented covariance structures:

  * Scaled Identity (SI)
  * Diagonal (DIAG)
  * Autoregressive (AR)
  * Heterogeneous Autoregressive (ARH)
  * Compound Symmetry (CS)
  * Heterogeneous Compound Symmetry (CSH)
  * Autoregressive Moving Average (ARMA)

### Usage

#### Model construction

`LMM(model, data; subject = nothing,  random = nothing, repeated = nothing)`

where:

* `model` is a fixed-effect model (`@formula`), example: `@formula(var ~ sequence + period + formulation)`

* `random` vector of random effects or single random effect. Effect can be declared like this: `VarEffect(@covstr(formulation), CSH)`. `@covstr` is a effect model: `@covstr(formulation)`. `CSH` is a  CovarianceType structure. Premade constants: SI, DIAG, AR, ARH, CS, CSH, ARMA.

* `repeated` is a repeated effect (only single).

* `subject` is a block-diagonal factor.

#### Fitting

```
fit!(lmm::LMM{T};
    solver::Symbol = :default,
    verbose = :auto,
    varlinkf = :exp,
    rholinkf = :sigm,
    aifirst::Bool = false,
    g_tol::Float64 = 1e-12,
    x_tol::Float64 = 1e-12,
    f_tol::Float64 = 1e-12,
    hcalck::Bool   = true,
    init = nothing,
    io::IO = stdout)
```

where:

* `solver` - :default / :nlopt / :cuda

* `verbose` - :auto / 1 / 2 / 3

* `varlinkf` - not implemented

* `rholinkf` - not implemented

* `g_tol` - absolute tolerance in the gradient

* `x_tol` - absolute tolerance of theta vector

* `f_tol` - absolute tolerance in changes of the REML

* `hcalck` - calculate REML Hessian

* `init` - initial theta values

* `io` - uotput IO

```@contents
Pages = [
        "examples.md",
        "details.md",
        "api.md"]
Depth = 3
```

See also:

[MixedModels.jl](https://github.com/JuliaStats/MixedModels.jl)
