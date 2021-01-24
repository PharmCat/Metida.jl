## Validation

### sleepstudy.csv

| Model  |  REML  Metida | REML SPSS|
|--------|--------|-------|
| 1 | 1729.4925602367025 | 1729.492560 |
| 2 | 1904.3265170722132 | 1904.327 |
| 3 | 1772.0953251997046 | 1772.095 |
| 4 | 1730.1895427398322 | 1730.189543 |


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

### Pastes.csv

| Model  |  REML  Metida | REML SPSS|
|--------|--------|-------|
| 5 | 246.99074585348623 | 246.990746 |

#### Model 5

```
lmm = Metida.LMM(@formula(strength~1), df;
random = [Metida.VarEffect(Metida.SI, subj = :batch), Metida.VarEffect(Metida.SI, subj = [:batch,  :cask])]
)
Metida.fit!(lmm)
```

```
```

### Reference dataset

* Schütz, H., Labes, D., Tomashevskiy, M. et al. Reference Datasets for Studies in a Replicate Design Intended for Average Bioequivalence with Expanding Limits. AAPS J 22, 44 (2020). https://doi.org/10.1208/s12248-020-0427-6

* Gregory Belenky, Nancy J. Wesensten, David R. Thorne, Maria L. Thomas, Helen C. Sing, Daniel P. Redmond, Michael B. Russo and Thomas J. Balkin (2003) Patterns of performance degradation and restoration during sleep restriction and subsequent recovery: a sleep dose-response study. Journal of Sleep Research 12, 1–12.

* O.L. Davies and P.L. Goldsmith (eds), Statistical Methods in Research and Production, 4th ed., Oliver and Boyd, (1972), section 6.6

* O.L. Davies and P.L. Goldsmith (eds), Statistical Methods in Research and Production, 4th ed., Oliver and Boyd, (1972), section 6.5

https://vincentarelbundock.github.io/Rdatasets/datasets.html


Stock, J.H. and Watson, M.W. (2007). Introduction to Econometrics, 2nd ed. Boston: Addison Wesley.
Anglin, P.M. and R. Gencay (1996) “Semiparametric estimation of a hedonic price function”, Journal of Applied Econometrics, 11(6), 633-648.
Anglin, P., and Gencay, R. (1996). Semiparametric Estimation of a Hedonic Price Function. Journal of Applied Econometrics, 11, 633–648.

Verbeek, M. (2004). A Guide to Modern Econometrics, 2nd ed. Chichester, UK: John Wiley.
O'Brien, R. G., and Kaiser, M. K. (1985) MANOVA method for analyzing repeated measures designs: An extensive primer. Psychological Bulletin 97, 316–333, Table 7.
