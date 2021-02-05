## Validation

Validation provided with 3 sections:
* REML validation for public datasets with Metida & SPSS
* Parameters validation for public datasets Metida & SPSS & MixedModels
* Validation with bioequivalence datasets with Metida & SPSS

## Section 1: REML validation for public datasets Metida vs SPSS

### REML result table

| Model  | DataSet |Used cov. types | REML  Metida | REML SPSS|
|--------|--------|--------|--------|-------|
| 1 | sleepstudy.csv | SI/SI | 1729.4925602367025 | 1729.492560 |
| 2 | sleepstudy.csv | CS/SI | 1904.3265170722132 | 1904.327 |
| 3 | sleepstudy.csv | CSH/SI | 1772.0953251997046 | 1772.095 |
| 4 | sleepstudy.csv | ARH/SI | 1730.1895427398322 | 1730.189543 |
| 5 | pastes.csv | SI,SI/SI | 246.99074585348623 | 246.990746 |
| 6 | pastes.csv | ARMA/SI | 246.81895071012508 | 246.818951 |
| 7 | penicillin.csv | SI,SI/SI | 330.86058899109184 | 330.860589 |

### sleepstudy.csv

#### Model 1

```
lmm = Metida.LMM(@formula(Reaction~Days), df;
  random = Metida.VarEffect(Metida.SI),
  subject = :Subject
  )
  Metida.fit!(lmm)
```

```
```
#### Model 2

```
lmm = Metida.LMM(@formula(Reaction~1), df;
  random = Metida.VarEffect(Metida.@covstr(Days), Metida.CS),
  subject = :Subject
  )
  Metida.fit!(lmm)
```

```
```

#### Model 3

```
lmm = Metida.LMM(@formula(Reaction~1), df;
  random = Metida.VarEffect(Metida.@covstr(Days), Metida.CSH, subj = :Subject)
  )
  Metida.fit!(lmm)
```

```
```

#### Model 4

```
lmm = Metida.LMM(@formula(Reaction~1), df;
  random = Metida.VarEffect(Metida.@covstr(Days), Metida.ARH, subj = :Subject)
  )
  Metida.fit!(lmm)
```

```
```

### pastes.csv

#### Model 5

```
lmm = Metida.LMM(@formula(strength~1), df;
random = [Metida.VarEffect(Metida.SI, subj = :batch), Metida.VarEffect(Metida.SI, subj = [:batch,  :cask])]
)
Metida.fit!(lmm)
```

```
```

#### Model 6

```
lmm = Metida.LMM(@formula(strength~1), df;
random = Metida.VarEffect(Metida.@covstr(cask), Metida.ARMA, subj = :batch),
)
Metida.fit!(lmm)
```

```
```

### penicillin.csv

#### Model 7

```
lmm = Metida.LMM(@formula(diameter~1), df;
random = [Metida.VarEffect(Metida.SI, subj = :plate), Metida.VarEffect(Metida.SI, subj = :sample)]
)
Metida.fit!(lmm)
```

```
```

## Section 2: Parameters validation for public datasets Metida & SPSS & MixedModels

not done yet

## Section 3: Validation with bioequivalence datasets with Metida & SPSS

not done yet

## Discussion

Optimization of REML function can depend on many factors. Most of all in some cases covariance parameters can be correlated (ill-conditioned/singular covariance matrix). So hypersurface in the maximum area can be very flat, that why the result can be different for different starting values (or for different software even REML is near equal). Also, some models can not be fitted for specific data at all. If the model not fitted try to check how meaningful and reasonable is the model or try to guess more robust initial conditions.
