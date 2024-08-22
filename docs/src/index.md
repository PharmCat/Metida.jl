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
  * Toeplitz (TOEP)
  * Toeplitz Parameterized (TOEPP)
  * Heterogeneous Toeplitz (TOEPH)
  * Heterogeneous Toeplitz Parameterized (TOEPHP)
  * Spatial Exponential (SPEXP)
  * Spatial Power (SPPOW)
  * Spatial Gaussian (SPGAU)
  * Unstructured (UN) 
  * Custom Covariance Type

## Limitations

  * Maximum length of block more than ~400 (observation-per-subject).
  * Observation number more than 160 000.
  * For MetidaCu number of blocks more than 40  (maximum length of block more than 4000).

Actually Metida can fit datasets with wore than  160k observation and 40k subjects levels on PC with 64 GB RAM. This is not "hard-coded" limitation, but depends on your model and data structure. Fitting of big datasets can take a lot of time. Optimal dataset size is less than 100k observations with maximum length of block less than 400.

!!! warning

    Julia v1.8 or higher required.

## Contents

```@contents
Pages = [
        "details.md",
        "examples.md",
        "validation.md",
        "api.md"]
Depth = 3
```

See also:

* [MixedModels.jl](https://github.com/JuliaStats/MixedModels.jl)

* [GLM.jl](https://github.com/JuliaStats/GLM.jl)

* [SweepOperator.jl](https://github.com/joshday/SweepOperator.jl)

## Reference

* Gelman, A.; Hill, J. (2007). Data Analysis Using Regression and Multilevel/Hierarchical Models. New York: Cambridge University Press. pp. 235–299. ISBN 978-0-521-68689-1.
