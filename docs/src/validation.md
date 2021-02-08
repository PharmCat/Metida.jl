# Validation

Validation provided with 3 sections:
* REML validation for public datasets with Metida & SPSS
* Parameters validation for public datasets Metida & SPSS & MixedModels
* Validation with bioequivalence datasets with Metida & SPSS

## Section 1: REML validation for public datasets Metida & SPSS

#### REML result table

| Model  | DataSet |Used cov. types | REML  Metida | REML SPSS|
|--------|--------|--------|--------|-------|
| 1 | sleepstudy.csv | SI/SI | 1729.4925602367025 | 1729.492560 |
| 2 | sleepstudy.csv | CS/SI | 1904.3265170722132 | 1904.327 |
| 3 | sleepstudy.csv | CSH/SI | 1772.0953251997046 | 1772.095 |
| 4 | sleepstudy.csv | ARH/SI | 1730.1895427398322 | 1730.189543 |
| 5 | Pastes.csv | SI,SI/SI | 246.99074585348623 | 246.990746 |
| 6 | Pastes.csv | ARMA/SI | 246.81895071012508 | 246.818951 |
| 7 | Penicillin.csv | SI,SI/SI | 330.86058899109184 | 330.860589 |
| 8 | RepeatedPulse.csv | SI/AR | 453.3395435627574 | 453.339544 |
| 9 | RepeatedPulse.csv | 0/AR | 471.85107712169827 | 471.851077 |
| 10 | RepeatedPulse.csv | AR/SI | 453.3395560121246 | 453.339555 |

#### sleepstudy.csv

##### Model 1

```
lmm = Metida.LMM(@formula(Reaction~Days), df;
  random = Metida.VarEffect(Metida.SI),
  subject = :Subject
  )
  Metida.fit!(lmm)
```

SPSS:
```
MIXED Reaction BY Days
  /CRITERIA=DFMETHOD(SATTERTHWAITE) CIN(95) MXITER(100) MXSTEP(10) SCORING(1)
    SINGULAR(0.000000000001) HCONVERGE(0, ABSOLUTE) LCONVERGE(0, ABSOLUTE) PCONVERGE(0.000001, ABSOLUTE)
  /FIXED=Days | SSTYPE(3)
  /METHOD=REML
  /RANDOM=INTERCEPT | SUBJECT(Subject) COVTYPE(ID).
```
##### Model 2

```
lmm = Metida.LMM(@formula(Reaction~1), df;
  random = Metida.VarEffect(Metida.@covstr(Days), Metida.CS),
  subject = :Subject
  )
  Metida.fit!(lmm)
```

SPSS:
```
MIXED Reaction BY Days
  /CRITERIA=DFMETHOD(SATTERTHWAITE) CIN(95) MXITER(100) MXSTEP(10) SCORING(1)
    SINGULAR(0.000000000001) HCONVERGE(0, ABSOLUTE) LCONVERGE(0, ABSOLUTE) PCONVERGE(0.000001, ABSOLUTE)
  /FIXED=INTERCEPT | SSTYPE(3)
  /METHOD=REML
  /RANDOM=Days | SUBJECT(Subject) COVTYPE(CS).
```

##### Model 3

```
lmm = Metida.LMM(@formula(Reaction~1), df;
  random = Metida.VarEffect(Metida.@covstr(Days), Metida.CSH, subj = :Subject)
  )
  Metida.fit!(lmm)
```

SPSS:
```
MIXED Reaction BY Days
  /CRITERIA=DFMETHOD(SATTERTHWAITE) CIN(95) MXITER(100) MXSTEP(10) SCORING(1)
    SINGULAR(0.000000000001) HCONVERGE(0, ABSOLUTE) LCONVERGE(0, ABSOLUTE) PCONVERGE(0.000001, ABSOLUTE)
  /FIXED=INTERCEPT | SSTYPE(3)
  /METHOD=REML
  /RANDOM=Days | SUBJECT(Subject) COVTYPE(CSH).
```

##### Model 4

```
lmm = Metida.LMM(@formula(Reaction~1), df;
  random = Metida.VarEffect(Metida.@covstr(Days), Metida.ARH, subj = :Subject)
  )
  Metida.fit!(lmm)
```

SPSS:
```
MIXED Reaction BY Days
  /CRITERIA=DFMETHOD(SATTERTHWAITE) CIN(95) MXITER(100) MXSTEP(10) SCORING(1)
    SINGULAR(0.000000000001) HCONVERGE(0, ABSOLUTE) LCONVERGE(0, ABSOLUTE) PCONVERGE(0.000001, ABSOLUTE)
  /FIXED=INTERCEPT | SSTYPE(3)
  /METHOD=REML
  /RANDOM=Days | SUBJECT(Subject) COVTYPE(ARH1).
```

#### pastes.csv

##### Model 5

```
lmm = Metida.LMM(@formula(strength~1), df;
random = [Metida.VarEffect(Metida.SI, subj = :batch), Metida.VarEffect(Metida.SI, subj = [:batch,  :cask])]
)
Metida.fit!(lmm)
```

SPSS:
```
MIXED strength
  /CRITERIA=DFMETHOD(SATTERTHWAITE) CIN(95) MXITER(100) MXSTEP(10) SCORING(1)
    SINGULAR(0.000000000001) HCONVERGE(0, ABSOLUTE) LCONVERGE(0, ABSOLUTE) PCONVERGE(0.000001, ABSOLUTE)
  /FIXED=| SSTYPE(3)
  /METHOD=REML
  /RANDOM=INTERCEPT | SUBJECT(batch) COVTYPE(ID)
  /RANDOM=INTERCEPT | SUBJECT(cask * batch) COVTYPE(ID).
```

##### Model 6

```
lmm = Metida.LMM(@formula(strength~1), df;
random = Metida.VarEffect(Metida.@covstr(cask), Metida.ARMA, subj = :batch),
)
Metida.fit!(lmm)
```

SPSS:
```
MIXED strength by cask
  /CRITERIA=DFMETHOD(SATTERTHWAITE) CIN(95) MXITER(100) MXSTEP(10) SCORING(1)
    SINGULAR(0.000000000001) HCONVERGE(0, ABSOLUTE) LCONVERGE(0, ABSOLUTE) PCONVERGE(0.000001, ABSOLUTE)
  /FIXED=| SSTYPE(3)
  /METHOD=REML
  /RANDOM=cask | SUBJECT(batch) COVTYPE(ARMA11).
```

#### penicillin.csv

##### Model 7

```
lmm = Metida.LMM(@formula(diameter~1), df;
random = [Metida.VarEffect(Metida.SI, subj = :plate), Metida.VarEffect(Metida.SI, subj = :sample)]
)
Metida.fit!(lmm)
```

SPSS:
```
MIXED diameter
  /CRITERIA=DFMETHOD(SATTERTHWAITE) CIN(95) MXITER(100) MXSTEP(10) SCORING(1)
    SINGULAR(0.000000000001) HCONVERGE(0, ABSOLUTE) LCONVERGE(0, ABSOLUTE) PCONVERGE(0.000001, ABSOLUTE)
  /FIXED=| SSTYPE(3)
  /METHOD=REML
  /RANDOM=INTERCEPT | SUBJECT(plate) COVTYPE(ID)
  /RANDOM=INTERCEPT | SUBJECT(sample) COVTYPE(ID).
```

#### RepeatedPulse.csv

##### Model 8

```
sort!(df, :Day)
lmm = Metida.LMM(@formula(Pulse~1), df;
random = Metida.VarEffect(Metida.@covstr(Time), Metida.SI),
repeated = Metida.VarEffect(Metida.@covstr(Day), Metida.AR),
subject = :Time
)
Metida.fit!(lmm)
```

SPSS:
```

```

##### Model 9

```
sort!(df, :Day)
lmm = Metida.LMM(@formula(Pulse~1), df;
repeated = Metida.VarEffect(Metida.@covstr(Day), Metida.AR),
subject = :Time
)
Metida.fit!(lmm)
```

SPSS:
```

```

##### Model 10

```
sort!(df, :Day)
lmm = Metida.LMM(@formula(Pulse~1), df;
random = Metida.VarEffect(Metida.@covstr(Day), Metida.AR),
subject = :Time
)
Metida.fit!(lmm)
```

SPSS:
```

```

## Section 2: Parameters validation for public datasets Metida & SPSS & MixedModels

not done yet

## Section 3: Validation with bioequivalence datasets with Metida & SPSS

not done yet

#### Model BE1

```
lmm = Metida.LMM(@formula(lnpk~sequence+period+treatment), dfrds;
    random = Metida.VarEffect(Metida.@covstr(1), Metida.SI),
    subject = :subject
    )
    Metida.fit!(lmm)
```

#### Model BE2

```
lmm = Metida.LMM(@formula(lnpk~sequence+period+treatment), dfrds;
    random = Metida.VarEffect(Metida.@covstr(treatment), Metida.CSH),
    repeated = Metida.VarEffect(Metida.@covstr(treatment), Metida.DIAG),
    subject = :subject
    )
    Metida.fit!(lmm)
```

#### Typical SPSS code

```
MIXED lnpk BY period sequence treatment subject
  /CRITERIA=DFMETHOD(SATTERTHWAITE) CIN(95) MXITER(100) MXSTEP(10) SCORING(1)
    SINGULAR(0.000000000001) HCONVERGE(0, ABSOLUTE) LCONVERGE(0, ABSOLUTE) PCONVERGE(0.000001, ABSOLUTE)
  /FIXED=period sequence treatment | SSTYPE(3)
  /METHOD=REML
  /RANDOM= subject(sequence) | COVTYPE(ID)
  /EMMEANS=TABLES(treatment) COMPARE REFCAT(FIRST) ADJ(LSD).

MIXED lnpk BY period treatment sequence subject
  /CRITERIA=CIN(90) MXITER(200) MXSTEP(20) SCORING(2) SINGULAR(0.000000000001) HCONVERGE(0,
    RELATIVE) LCONVERGE(0.0000000000001, RELATIVE) PCONVERGE(0, RELATIVE)
  /FIXED=period treatment sequence | SSTYPE(3)
  /METHOD=REML
  /RANDOM=treatment | SUBJECT(subject) COVTYPE(CSH)
  /REPEATED=treatment | SUBJECT(subject*period) COVTYPE(DIAG)
  /EMMEANS=TABLES(treatment) COMPARE REFCAT(FIRST) ADJ(LSD).
```

Full SPSS code provided in validation folder ([here](https://github.com/PharmCat/Metida.jl/blob/master/validation/spssrdscode.sps.txt)).

Validation dataset available [here](https://link.springer.com/article/10.1208%2Fs12248-020-0427-6), [12248_2020_427_MOESM2_ESM.xls](https://static-content.springer.com/esm/art%3A10.1208%2Fs12248-020-0427-6/MediaObjects/12248_2020_427_MOESM2_ESM.xls).

## Discussion

Optimization of REML function can depend on many factors. Most of all in some cases covariance parameters can be correlated (ill-conditioned/singular covariance matrix). So hypersurface in the maximum area can be very flat, that why the result can be different for different starting values (or for different software even REML is near equal). Also, some models can not be fitted for specific data at all. If the model not fitted try to check how meaningful and reasonable is the model or try to guess more robust initial conditions.
