### Example 1 - Continuous and categorical predictors

```@example 1
using Metida, StatsPlots, CSV, DataFrames, MixedModels;

rds = CSV.File(joinpath(dirname(pathof(Metida)), "..", "test", "csv",  "1fptime.csv"); types = [String, String, Float64, Float64]) |> DataFrame

@df rds plot(:time, :response, group = (:subject, :factor), colour = [:red :blue], legend = false)
png("plot1.png"); nothing # hide
```

![](plot1.png)

Metida result:

```@example 1
lmm = Metida.LMM(@formula(response ~1 + factor*time), rds;
random = Metida.VarEffect(Metida.@covstr(1 + time), Metida.CSH),
subject = [:subject, :factor]
)
Metida.fit!(lmm)
```

MixedModels result:

```@example 1
fm = @formula(response ~ 1 + factor*time + (1 + time|subject&factor))
mm = fit(MixedModel, fm, rds, REML=true)
```

### Example 2 - Two random factors (Penicillin data)

Metida:

```@example 1

df          = CSV.File(joinpath(dirname(pathof(Metida)), "..", "test", "csv", "Penicillin.csv"); types = [String, Float64, String, String]) |> DataFrame
df.diameter = float.(df.diameter)

lmm = Metida.LMM(@formula(diameter ~ 1), df;
random = [Metida.VarEffect(Metida.SI, subj = :plate), Metida.VarEffect(Metida.SI, subj = :sample)]
)
Metida.fit!(lmm)
```

MixedModels:

```@example 1

fm2 = @formula(diameter ~ 1 + (1|plate) + (1|sample))
mm = fit(MixedModel, fm2, df, REML=true)
```

### Example 3 - Repeated ARMA/AR/ARH

```@example 1
rds = CSV.File(joinpath(dirname(pathof(Metida)), "..", "test", "csv",  "1freparma.csv"); types = [String, String, Float64, Float64]) |> DataFrame

@df rds plot(:time, :response, group = (:subject, :factor), colour = [:red :blue], legend = false)
png("plot2.png"); nothing # hide
```

![](plot2.png)

ARMA:

```@example 1
lmm = Metida.LMM(@formula(response ~ 1 + factor*time), rds;
random = Metida.VarEffect(Metida.@covstr(factor), Metida.DIAG),
repeated = Metida.VarEffect(Metida.ARMA),
subject = [:subject, :factor]
)
Metida.fit!(lmm)
```

AR:

```@example 1
lmm = Metida.LMM(@formula(response ~ 1 + factor*time), rds;
random = Metida.VarEffect(Metida.@covstr(factor), Metida.DIAG),
repeated = Metida.VarEffect(Metida.AR),
subject = [:subject, :factor]
)
Metida.fit!(lmm)
```

ARH:

```@example 1
lmm = Metida.LMM(@formula(response ~ 1 + factor*time), rds;
random = Metida.VarEffect(Metida.@covstr(factor), Metida.DIAG),
repeated = Metida.VarEffect(Metida.ARH),
subject = [:subject, :factor]
)
Metida.fit!(lmm)
```

### Example 4

#### Model 1

```
df0 = CSV.File(joinpath(dirname(pathof(Metida)), "..", "test", "csv", "df0.csv")) |> DataFrame

lmm = LMM(@formula(var ~ sequence + period + formulation), df0;
random   = VarEffect(@covstr(formulation), CSH),
repeated = VarEffect(@covstr(formulation), DIAG),
subject  = :subject)
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

#### Model 2

```
lmm = LMM(
    @formula(var ~ sequence + period + formulation), df0;
    random   = VarEffect(@covstr(formulation), SI),
    repeated = VarEffect(@covstr(formulation), DIAG),
    subject  = :subject,
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

#### Model 3

```
lmm = LMM(@formula(var ~ sequence + period + formulation), df0;
    random = VarEffect(@covstr(subject), SI)
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
