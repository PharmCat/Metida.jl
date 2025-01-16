## Example 1 - Continuous and categorical predictors

```@example lmmexample
using Metida, CSV, DataFrames, CategoricalArrays, MixedModels;

rds = CSV.File(joinpath(dirname(pathof(Metida)), "..", "test", "csv",  "1fptime.csv"); types = [String, String, Float64, Float64]) |> DataFrame

rds2 = CSV.File(joinpath(dirname(pathof(Metida)), "..", "test", "csv",  "ftdf3.csv"); types = [String,  Float64, Float64, String, String, String, String, String, Float64]) |> DataFrame


devday = CSV.File(joinpath(dirname(pathof(Metida)), "..", "test", "csv",  "devday.csv"); types = [Float64, String, String, String ]) |> DataFrame

nothing; # hide
```

![](plot1.png)

Metida result:

```@example lmmexample
lmm = LMM(@formula(response ~1 + factor*time), rds;
random = VarEffect(@covstr(1 + time|subject&factor), CSH),
)
fit!(lmm)
```

MixedModels result:

```@example lmmexample
fm = @formula(response ~ 1 + factor*time + (1 + time|subject&factor))
mm = fit(MixedModel, fm, rds, REML=true)
println(mm) #hide
```

## Example 2 - Two random factors (Penicillin data)

Metida:

```@example lmmexample

df          = CSV.File(joinpath(dirname(pathof(Metida)), "..", "test", "csv", "Penicillin.csv"); types = [String, Float64, String, String]) |> DataFrame
df.diameter = float.(df.diameter)

lmm = LMM(@formula(diameter ~ 1), df;
random = [VarEffect(@covstr(1|plate), SI), VarEffect(@covstr(1|sample), SI)]
)
fit!(lmm)
```

MixedModels:

```@example lmmexample

fm2 = @formula(diameter ~ 1 + (1|plate) + (1|sample))
mm = fit(MixedModel, fm2, df, REML=true)
println(mm) #hide
```

## Example 3 - Repeated ARMA/AR/ARH

```@example lmmexample
rds = CSV.File(joinpath(dirname(pathof(Metida)), "..", "test", "csv",  "1freparma.csv"); types = [String, String, Float64, Float64]) |> DataFrame

nothing # hide
```

![](plot2.png)

ARMA:

```@example lmmexample
lmm = LMM(@formula(response ~ 1 + factor*time), rds;
random = VarEffect(@covstr(factor|subject&factor), DIAG),
repeated = VarEffect(@covstr(1|subject&factor), ARMA),
)
fit!(lmm)
```

AR:

```@example lmmexample
lmm = Metida.LMM(@formula(response ~ 1 + factor*time), rds;
random = VarEffect(@covstr(factor|subject&factor), DIAG),
repeated = VarEffect(@covstr(1|subject&factor), AR),
)
fit!(lmm)
```

ARH:

```@example lmmexample
lmm = Metida.LMM(@formula(response ~ 1 + factor*time), rds;
random = VarEffect(@covstr(factor|subject&factor), DIAG),
repeated = VarEffect(@covstr(1|subject&factor), ARH),
)
fit!(lmm)
```

## Example 4 - SAS relation

### Model 1

```
df0 = CSV.File(joinpath(dirname(pathof(Metida)), "..", "test", "csv", "df0.csv")) |> DataFrame

lmm = LMM(@formula(var ~ sequence + period + formulation), df0;
random   = VarEffect(@covstr(formulation|subject), CSH),
repeated = VarEffect(@covstr(formulation|subject), DIAG),
)
fit!(lmm)
```

SAS code:

```
PROC MIXED data=df0;
CLASSES subject sequence period formulation;
MODEL  var = sequence period formulation/ DDFM=SATTERTH s;
RANDOM  formulation/TYPE=CSH SUB=subject G V;
REPEATED/GRP=formulation SUB=subject R;
RUN;
```

### Model 2

```
lmm = LMM(
    @formula(var ~ sequence + period + formulation), df0;
    random   = VarEffect(@covstr(formulation|subject), SI),
    repeated = VarEffect(@covstr(formulation|subject), DIAG),
)
fit!(lmm)
```

SAS code:

```
PROC MIXED data=df0;
CLASSES subject sequence period formulation;
MODEL  var = sequence period formulation/ DDFM=SATTERTH s;
RANDOM  formulation/TYPE=VC SUB=subject G V;
REPEATED/GRP=formulation SUB=subject R;
RUN;
```

### Model 3

```
lmm = LMM(@formula(var ~ sequence + period + formulation), df0;
    random = VarEffect(@covstr(subject|1), SI)
    )
fit!(lmm)
```

SAS code:

```
PROC MIXED data=df0;
CLASSES subject sequence period formulation;
MODEL  var = sequence period formulation/ DDFM=SATTERTH s;
RANDOM  subject/TYPE=VC G V;
RUN;
```

## Example 5 - Working with Effects.jl

```
using Effects, StatsModels

lmm = LMM(@formula(var ~ sequence + period + formulation), df0;
    random = VarEffect(@covstr(subject|1), SI)
    )
fit!(lmm)

table_model = StatsModels.TableRegressionModel(lmm, lmm.mf, lmm.mm)

emmeans(tm)

effects(Dict(:period => ["1", "2", "3", "4"]), tm)
```


## Unstructured covariance 

Unstructured covariance example.


Metida result:

```@example lmmexample
lmm = Metida.LMM(@formula(response~factor), rds2;
    random = Metida.VarEffect(Metida.@covstr(r1|subject), UN),
    )
Metida.fit!(lmm)
```

MixedModels result:

```@example lmmexample
mm = fit(MixedModel, @formula(response ~ factor+ (0+r1|subject)), rds2, REML = true)
println(mm) #hide
```

## Augmented covariance (Experimental)

Covariance modificator `ACOV()` can be used as second repeated effect. In this case covariance calculated with existed matrix, 
that was build at previous step. For example, addition `ACOV(AR)` to `DIAG` structure is the same as `ARH` if same blocking factor used.

```@example lmmexample
   lmm1 = Metida.LMM(@formula(response ~ 1), rds2;
    repeated = [Metida.VarEffect(Metida.@covstr(r1|subject), Metida.DIAG), Metida.VarEffect(Metida.@covstr(1|subject), Metida.ACOV(Metida.AR))]
    )
    Metida.fit!(lmm1)
```

```@example lmmexample
    lmm2 = Metida.LMM(@formula(response ~ 1), rds2;
    repeated = [Metida.VarEffect(Metida.@covstr(r1|subject), Metida.ARH)]
    )
    Metida.fit!(lmm2)
```

R-part of variance-covariance matrix:

```@example lmmexample
Metida.rmatrix(lmm1, 1)
```

If nested blocking factor used - covariance modification applyed only within that blocks (R-part of variance-covariance matrix is the same):

```@example lmmexample
   lmm = Metida.LMM(@formula(response ~ 1), rds2;
    repeated = [Metida.VarEffect(Metida.@covstr(r1|subject), Metida.DIAG), Metida.VarEffect(Metida.@covstr(1|subject), Metida.ACOV(Metida.AR))]
    )
Metida.fit!(lmm)
Metida.rmatrix(lmm, 1)
```

For nested models covariance structure can be expanded as follows:
* the first layer describes unstructured the device-device covariance;
* the second layer adds the time covariance for each device

```@example lmmexample
lmm = Metida.LMM(@formula(resp ~ 0 + device), devday;
    repeated = [Metida.VarEffect(Metida.@covstr(device|subj&day), Metida.UN), 
    Metida.VarEffect(Metida.@covstr(1|subj&device), Metida.ACOV(Metida.AR))]
    )
    Metida.fit!(lmm)
```
R-matrix:

```@example lmmexample
Metida.rmatrix(lmm, 1)
```

