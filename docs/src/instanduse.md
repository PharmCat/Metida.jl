### Installation

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

    Check that all categorical variables is categorical.


```@example lmmexample
categorical!(df, :subject);
categorical!(df, :period);
categorical!(df, :sequence);
categorical!(df, :formulation);
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

#### Step 1: make CustomCovarianceStruct

```@docs
Metida.CustomCovarianceStruct
```

#### Step 2: make CustomCovarianceType

```@docs
Metida.CustomCovarianceType
```

#### Step 3 Fit your model

```@example lmmexample
using Metida, CSV, DataFrames # hide

df0 = CSV.File(joinpath(dirname(pathof(Metida)), "..", "test", "csv",  "df0.csv"); types = [String, String, String, String, Float64, Float64]) |> DataFrame

#Make struct for G and R matrix
ccsg = CustomCovarianceStruct((q,p) -> (q, 1), Metida.gmat_csh!)
ccsr = CustomCovarianceStruct((q,p) -> (q, 0), Metida.rmatp_diag!)

#Make type struct
CCTG = CustomCovarianceType(ccsg)
CCTR = CustomCovarianceType(ccsr)

#Make model
lmm = LMM(@formula(var~sequence+period+formulation), df0;
random = VarEffect(@covstr(formulation|subject), CCTG),
repeated = VarEffect(@covstr(formulation|subject), CCTR),
)

#Fit model
fit!(lmm)
```
