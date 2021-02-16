### Installation

```
import Pkg; Pkg.add("Metida")
```

### Simple example

#### Step 1: Load data

Load provided data with CSV and DataFrames:

```
using CSV, DataFrames
df = CSV.File(joinpath(dirname(pathof(Metida)), "..", "test", "csv", "df0.csv")) |> DataFrame
```

Check that all categorical variables is categorical.

```
categorical!(df, :subject);
categorical!(df, :period);
categorical!(df, :sequence);
categorical!(df, :formulation);
```

#### Step 2: Make model

```
lmm = LMM(@formula(var~sequence+period+formulation), df;
random = VarEffect(@covstr(formulation|subject), CSH),
repeated = VarEffect(@covstr(formulation|subject), DIAG))
```

#### Step 3: Fit

```
fit!(lmm)
```

### Model construction

[`LMM`](@ref)

* `model` - example: `@formula(var ~ sequence + period + formulation)`

* `random` - effects can be specified like this: `VarEffect(@covstr(formulation|subject), CSH)`. `@covstr` is a effect model: `@covstr(formulation|subject)`. `CSH` is a  CovarianceType structure. Premade constants: SI, DIAG, AR, ARH, CS, CSH, ARMA. If not specified only repeated used.

* `repeated` - can be specified like random effect. If not specified `VarEffect(@covstr(1|1), SI)` used. If no repeated effects specified vector of ones used.


### Fitting

[`fit!`](@ref)

* `solver` - `:default` solving with Optim.jl, for `:nlopt` and `:cuda` MetidaNLopt.jl and MetidaCu.jl should be installed.

* `verbose` - 1 - only log,  2 - log and print,  3 - print only errors, other log, 0 (or any other value) - no logging.
