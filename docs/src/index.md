# Metida

```@meta
CurrentModule = Metida
```

## Mixed Models

Multilevel models (also known as hierarchical linear models, linear mixed-effect model, mixed models, nested data models, random coefficient, random-effects models, random parameter models, or split-plot designs) are statistical models of parameters that vary at more than one level. An example could be a model of student performance that contains measures for individual students as well as measures for classrooms within which the students are grouped. These models can be seen as generalizations of linear models (in particular, linear regression), although they can also extend to non-linear models. These models became much more popular after sufficient computing power and software became available. ([Wiki](https://en.wikipedia.org/wiki/Multilevel_model))


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

* `random` vector of random effects or single random effect. Effects can be specified like this: `VarEffect(@covstr(formulation), CSH)`. `@covstr` is a effect model: `@covstr(formulation)`. `CSH` is a  CovarianceType structure. Premade constants: SI, DIAG, AR, ARH, CS, CSH, ARMA.

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

* `rholinkf` - :sigm / :atan

* `aifirst` - first iteration with AI-like method

* `g_tol` - absolute tolerance in the gradient

* `x_tol` - absolute tolerance of theta vector

* `f_tol` - absolute tolerance in changes of the REML

* `hcalck` - calculate REML Hessian

* `init` - initial theta values

* `io` - uotput IO

### Contents

```@contents
Pages = [
        "details.md",
        "examples.md",
        "validation.md"
        "api.md"]
Depth = 3
```

See also:

* [MixedModels.jl](https://github.com/JuliaStats/MixedModels.jl)

* [GLM.jl](https://github.com/JuliaStats/GLM.jl)

* [SweepOperator.jl](https://github.com/joshday/SweepOperator.jl)

### Reference

* Gelman, A.; Hill, J. (2007). Data Analysis Using Regression and Multilevel/Hierarchical Models. New York: Cambridge University Press. pp. 235–299. ISBN 978-0-521-68689-1.
