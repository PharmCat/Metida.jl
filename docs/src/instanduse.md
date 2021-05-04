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
using Metida, CSV, DataFrames

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

```@example lmmexample
lmm = LMM(@formula(var~sequence+period+formulation), df;
random = VarEffect(@covstr(formulation|subject), CSH),
repeated = VarEffect(@covstr(formulation|subject), DIAG));
```

#### Step 3: Fit

```@example lmmexample
fit!(lmm)
```

##### Check warnings and errors in log.

```@example lmmexample
lmm.log
```

##### Type III Tests of Fixed Effects

!!! warning
    Experimental

```@example lmmexample
anova(lmm)
```

### Model construction

```@docs
Metida.LMM
```

* `model` - example: `@formula(var ~ sequence + period + formulation)`

* `random` - effects can be specified like this: `VarEffect(@covstr(formulation|subject), CSH)`. `@covstr` is a effect model: `@covstr(formulation|subject)`. `CSH` is a  CovarianceType structure. Premade constants: SI, DIAG, AR, ARH, CS, CSH, ARMA. If not specified only repeated used.

* `repeated` - can be specified like random effect. If not specified `VarEffect(@covstr(1|1), SI)` used. If no repeated effects specified vector of ones used.

### Random/repeated model

```@docs
Metida.@covstr
```

### Random/repeated effect construction

```@docs
Metida.VarEffect
```

### Fitting

```@docs
Metida.fit!
```

* `solver` - `:default` solving with Optim.jl, for `:nlopt` and `:cuda` MetidaNLopt.jl and MetidaCu.jl should be installed.

* `verbose` - 1 - only log,  2 - log and print,  3 - print only errors, other log, 0 (or any other value) - no logging.

### Custom structure

#### Step 1: make custom CovarianceMethod for R and G matrix

```@docs
Metida.CovmatMethod
```

#### Step 2: make custom CovarianceType

```
  CovarianceType(cm::AbstractCovmatMethod)
```

See: [`Metida.CovarianceType`](@ref)

#### Step 3 Fit your model

```@example lmmexample

#Make methods for G and R matrix and CovarianceType struct
CCTG = CovarianceType(CovmatMethod((q,p) -> (q, 1), Metida.gmat_csh!))
CCTR = CovarianceType(CovmatMethod((q,p) -> (q, 0), Metida.rmatp_diag!))

#Make model
lmm = LMM(@formula(var~sequence+period+formulation), df;
random = VarEffect(@covstr(formulation|subject), CCTG),
repeated = VarEffect(@covstr(formulation|subject), CCTR),
)

#Fit model
fit!(lmm)
```
