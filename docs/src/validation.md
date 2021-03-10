# Validation

Validation provided with 3 sections:
* REML validation for public datasets with Metida & SPSS
* Parameters validation for public datasets Metida & SPSS & MixedModels
* Validation with bioequivalence datasets with Metida & SPSS

To run validation:

```
using Metida; include(joinpath(dirname(pathof(Metida)), "..", "test", "validation.jl"))
```

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
lmm = LMM(@formula(Reaction~Days), df;
  random = VarEffect(@covstr(1|Subject), SI),
  )
  fit!(lmm)
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
lmm = LMM(@formula(Reaction~1), df;
  random = VarEffect(Metida.@covstr(Days|Subject), CS),
  )
  fit!(lmm)
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
lmm = LMM(@formula(Reaction~1), df;
  random = VarEffect(@covstr(Days|Subject), CSH)
  )
  fit!(lmm)
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
lmm = LMM(@formula(Reaction~1), df;
  random = VarEffect(@covstr(Days|Subject), ARH)
  )
   fit!(lmm)
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
lmm =  LMM(@formula(strength~1), df;
random = [VarEffect(@covstr(1|batch), SI),  VarEffect(@covstr(1|batch & cask), SI)]
)
fit!(lmm)
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
lmm =  LMM(@formula(strength~1), df;
random = VarEffect(Metida.@covstr(cask|batch),  ARMA),
)
fit!(lmm)
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
lmm =  LMM(@formula(diameter ~ 1), df;
random = [VarEffect(@covstr(1|plate), SI), VarEffect(@covstr(1|sample), SI)]
)
fit!(lmm)
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
lmm =  LMM(@formula(Pulse~1), df;
random =  VarEffect(Metida.@covstr(Time|Time),  SI),
repeated =  VarEffect(Metida.@covstr(Day|Time),  AR),
)
fit!(lmm)
```

SPSS:
```
MIXED Pulse BY Day Time
  /CRITERIA=DFMETHOD(SATTERTHWAITE) CIN(95) MXITER(100) MXSTEP(10) SCORING(1)
    SINGULAR(0.000000000001) HCONVERGE(0, ABSOLUTE) LCONVERGE(0, ABSOLUTE) PCONVERGE(0.000001, ABSOLUTE)
  /FIXED=| SSTYPE(3)
  /METHOD=REML
  /RANDOM=Time | SUBJECT(Time) COVTYPE(ID)
  /REPEATED = Day | SUBJCET(Time) COVTYPE(AR1).
```

##### Model 9

```
sort!(df, :Day)
lmm =  LMM(@formula(Pulse~1), df;
repeated = VarEffect(Metida.@covstr(Day|Time),  AR),
)
 fit!(lmm)
```

SPSS:
```
MIXED Pulse BY Day Time
  /CRITERIA=DFMETHOD(SATTERTHWAITE) CIN(95) MXITER(100) MXSTEP(10) SCORING(1)
    SINGULAR(0.000000000001) HCONVERGE(0, ABSOLUTE) LCONVERGE(0, ABSOLUTE) PCONVERGE(0.000001, ABSOLUTE)
  /FIXED=| SSTYPE(3)
  /METHOD=REML
  /REPEATED = Day | SUBJCET(Time) COVTYPE(AR1).
```

##### Model 10

```
sort!(df, :Day)
lmm =  LMM(@formula(Pulse~1), df;
random =  VarEffect(Metida.@covstr(Day|Time),  AR),
)
 fit!(lmm)
```

SPSS:
```
MIXED Pulse BY Day Time
  /CRITERIA=DFMETHOD(SATTERTHWAITE) CIN(95) MXITER(100) MXSTEP(10) SCORING(1)
    SINGULAR(0.000000000001) HCONVERGE(0, ABSOLUTE) LCONVERGE(0, ABSOLUTE) PCONVERGE(0.000001, ABSOLUTE)
  /FIXED=| SSTYPE(3)
  /METHOD=REML
  /RANDOM=Day | SUBJECT(Time) COVTYPE(AR1).
```

## Section 2: Parameters validation for public datasets Metida & SPSS & MixedModels

##### Model 7
Metida:
```
lmm = LMM(@formula(diameter ~ 1), df;
random = [VarEffect(@covstr(1|plate), SI), VarEffect(@covstr(1|sample), SI)]
)
fit!(lmm)
```

MixedModels:
```
fm = @formula(diameter ~ 1 + (1|plate) + (1|sample))
mm  = fit(MixedModel, fm, df, REML=true)
```

SPSS:
```
MIXED diameter
  /CRITERIA=DFMETHOD(SATTERTHWAITE) CIN(95) MXITER(100) MXSTEP(10) SCORING(1)
    SINGULAR(0.000000000001) HCONVERGE(0, ABSOLUTE) LCONVERGE(0, ABSOLUTE) PCONVERGE(0.000001, ABSOLUTE)
  /FIXED=| SSTYPE(3)
  /METHOD=REML
  /EMMEANS=TABLES(OVERALL)
  /RANDOM=INTERCEPT | SUBJECT(plate) COVTYPE(ID)
  /RANDOM=INTERCEPT | SUBJECT(sample) COVTYPE(ID).
```

| Model | Parameter  | Value Metida | Value MM | Value SPSS |
|--------|--------|--------|--------|-------|
| 7 | (Intercept) estimate | 22.9722 | 22.9722 | 22.972 |
| 7 | (Intercept) SE | 0.808573 | 0.808572 | 0.809 |
| 7 | plate   σ² | 0.716908 | 0.716908 | 0.716908 |
| 7 | sample   σ² | 3.73092 | 3.730901 | 3.730918 |
| 7 | Residual   σ² | 0.302415 | 0.302416 | 0.302415 |

## Section 3: Validation with bioequivalence datasets with Metida & SPSS

#### Model BE-B

```
lmm =  LMM(@formula(lnpk~sequence+period+treatment), dfrds;
random =  VarEffect(Metida.@covstr(1|subject),  SI),
)
fit!(lmm)
```

#### Model BE-C

```
lmm =  LMM(@formula(lnpk~sequence+period+treatment), dfrds;
    random =  VarEffect(Metida.@covstr(treatment|subject),  CSH),
    repeated =  VarEffect(Metida.@covstr(treatment|subject),  DIAG),
    )
     fit!(lmm)
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
#### Results

```@example lmmexample
using Metida; # hide
include(joinpath(dirname(pathof(Metida)), "..", "test", "validation_init.jl")); # hide
include(joinpath(dirname(pathof(Metida)), "..", "test", "validation_s3.jl"))
```
Full SPSS code provided in validation folder ([here](https://github.com/PharmCat/Metida.jl/blob/master/validation/spssrdscode.sps.txt)).

SPSS [output](https://github.com/PharmCat/Metida.jl/blob/master/validation/RDS-OUTPUT.docx) in DOCX format.

Validation dataset available [here](https://link.springer.com/article/10.1208%2Fs12248-020-0427-6), [12248_2020_427_MOESM2_ESM.xls](https://static-content.springer.com/esm/art%3A10.1208%2Fs12248-020-0427-6/MediaObjects/12248_2020_427_MOESM2_ESM.xls).

#### Validation report

Validation and report can be done on local machine with Weave.jl and Pandoc.

```julia
using Weave, Metida
weave(joinpath(dirname(pathof(Metida)), "..", "test", "validation_report.jmd");
doctype = "pandoc2pdf",
out_path = :pwd,
pandoc_options=["--toc", "-V colorlinks=true" , "-V linkcolor=blue", "-V urlcolor=red", "-V toccolor=gray"])
```   

Report will be saved in Julia working directory. For your own purpose you can edit validation_report.jmd template.
