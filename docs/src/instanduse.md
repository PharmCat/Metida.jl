### Installation

```@setup lmmexample
ENV["GKSwstype"] = "nul"
using Plots, StatsPlots, CSV, DataFrames, Metida

gr()

Plots.reset_defaults()

rds = CSV.File(joinpath(dirname(pathof(Metida)), "..", "test", "csv",  "1fptime.csv"); types = [String, String, Float64, Float64]) |> DataFrame

p = @df rds plot(:time, :response, group = (:subject, :factor), colour = [:red :blue], legend = false)

png(p, "plot1.png")

rds = CSV.File(joinpath(dirname(pathof(Metida)), "..", "test", "csv",  "1freparma.csv"); types = [String, String, Float64, Float64]) |> DataFrame

p = @df rds plot(:time, :response, group = (:subject, :factor), colour = [:red :blue], legend = false)

png(p, "plot2.png")
```

```
import Pkg; Pkg.add("Metida")
```

### Simple example

* [`LMM`](@ref)
* [`Metida.@covstr`](@ref)
* [`Metida.VarEffect`](@ref)
* [`fit!`](@ref)

#### Step 1: Load data

Load provided data with CSV and DataFrames:

```@example lmmexample
using Metida, CSV, DataFrames, CategoricalArrays

df = CSV.File(joinpath(dirname(pathof(Metida)), "..", "test", "csv", "df0.csv")) |> DataFrame;
nothing # hide
```

!!! note

    Check that all categorical variables are categorical.


```@example lmmexample
transform!(df, :subject => categorical, renamecols=false)
transform!(df, :period => categorical, renamecols=false)
transform!(df, :sequence => categorical, renamecols=false)
transform!(df, :formulation => categorical, renamecols=false)
nothing # hide
```

#### Step 2: Make model

Make model with `@formula` macro from `StatsModels`.
Define `random` and `repreated` effects with [`Metida.VarEffect`](@ref) using [`Metida.@covstr`](@ref) macros. Left side of `@covstr` is model of effect and
right side is a effect itself. [`Metida.HeterogeneousCompoundSymmetry`](@ref) and [`Metida.Diag`](@ref) (Diagonal) in example bellow is a model of variance-covariance structure. See also [`Metida.@lmmformula`](@ref) macro.

!!! note

    In some cases levels of repeated effect should not be equal inside each level of subject or model will not have any sense. For example, it is assumed that usually CSH or UN (Unstructured) using with levels of repeated effect is different inside each level of subject. Metida does not check this!


```@example lmmexample
lmm = LMM(@formula(var~sequence+period+formulation), df;
random = VarEffect(@covstr(formulation|subject), CSH),
repeated = VarEffect(@covstr(formulation|subject), DIAG));
```

Also [`Metida.@lmmformula`](@ref) macro can be used:

```julia
lmm = LMM(@lmmformula(var~sequence+period+formulation,
    random = formulation|subject:CSH,
    repeated = formulation|subject:DIAG),
    df)
```

#### Step 3: Fit

Just fit the model.

```@example lmmexample
fit!(lmm)
```

##### Check warnings and errors in log.

```@example lmmexample
lmm.log
```

#### Confidence intervals for coefficients


```@example lmmexample
confint(lmm)
```

!!! note

    Satterthwaite approximation for the denominator degrees of freedom used by default.

#### StatsBsae API

StatsBsae API implemented: [`Metida.islinear`](@ref), [`Metida.confint`](@ref), [`Metida.coef`](@ref), [`Metida.coefnames`](@ref), [`Metida.dof_residual`](@ref), [`Metida.dof`](@ref), [`Metida.loglikelihood`](@ref), [`Metida.aic`](@ref), [`Metida.bic`](@ref), [`Metida.aicc`](@ref),  [`Metida.isfitted`](@ref), [`Metida.vcov`](@ref), [`Metida.stderror`](@ref), [`Metida.modelmatrix`](@ref), [`Metida.response`](@ref), [`Metida.crossmodelmatrix`](@ref), [`Metida.coeftable`](@ref), [`Metida.responsename`](@ref)



##### Type III Tests of Fixed Effects

!!! warning
    Experimental

```@example lmmexample
typeiii(lmm)
```

### Model construction

To construct model you can use [`LMM`](@ref) constructor. 

* `model` - example: `@formula(var ~ sequence + period + formulation)`

* `random` - effects can be specified like this: `VarEffect(@covstr(formulation|subject), CSH)`. `@covstr` is a effect model: `@covstr(formulation|subject)`. `CSH` is a  CovarianceType structure. Premade constants: SI, DIAG, AR, ARH, CS, CSH, ARMA, TOEP, TOEPH, UN, ets. If not specified only repeated used.

* `repeated` - can be specified like random effect. If not specified `VarEffect(@covstr(1|1), SI)` used. If no repeated effects specified vector of ones used.
