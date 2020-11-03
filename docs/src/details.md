## Details

### V

```math
V_{i} = Z_{i}GZ_i'+R_{i}
```

### REML

```math
logREML(\theta,\beta) = -\frac{N-p}{2} - \frac{1}{2}\sum_{i=1}^nlog|V_{i}|-

-\frac{1}{2}log|\sum_{i=1}^nX_i'V_i^{-1}X_i|-\frac{1}{2}\sum_{i=1}^n(y_i - X_{i}\beta)'V_i^{-1}(y_i - X_{i}\beta)
```

### Beta

```math
\beta = {(\sum_{i=1}^n X_{i}'V_i^{-1}X_{i})}^{-1}(\sum_{i=1}^n X_{i}'V_i^{-1}y_{i})
```

### Sweep

Details see: https://github.com/joshday/SweepOperator.jl
