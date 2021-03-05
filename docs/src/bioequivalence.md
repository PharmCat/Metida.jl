## Type B

```@example lmmexample
using Metida, CSV, DataFrames
# example data
df = CSV.File(joinpath(dirname(pathof(Metida)), "..", "test", "csv", "df0.csv")) |> DataFrame;
transform!(df, :subject => categorical, renamecols=false)
transform!(df, :period => categorical, renamecols=false)
transform!(df, :sequence => categorical, renamecols=false)
transform!(df, :formulation => categorical, renamecols=false)

lmm = LMM(@formula(var~sequence+period+formulation), df;
random = VarEffect(@covstr(1|subject), SI),
)
fit!(lmm)
ci = confint(lmm)[end]
exp.(ci) .* 100.0
```

## Type C

```@example lmmexample
lmm =LMM(@formula(var~sequence+period+formulation), df;
random = VarEffect(@covstr(formulation|subject), CSH),
repeated = VarEffect(@covstr(formulation|subject), DIAG),
)
fit!(lmm)
ci = confint(lmm)[end]
exp.(ci) .* 100.0
```
