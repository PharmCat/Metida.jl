## Details

#### Model

In matrix notation a mixed effect model can be represented as:

```math
y = X\beta + Zu + \epsilon
```

#### V

```math
V_{i} = Z_{i}GZ_i'+R_{i}
```

#### Henderson's «mixed model equations»

```math
\begin{pmatrix}X'R^{-1}X&X'R^{-1}Z\\Z'R^{-1}X&Z'R^{-1}Z+G_{-1}\end{pmatrix}  \begin{pmatrix}\widehat{\beta} \\ \widehat{u} \end{pmatrix}= \begin{pmatrix}X'R^{-1}y\\Z'R^{-1}y\end{pmatrix}
```

The solution to the mixed model equations is a maximum likelihood estimate when the distribution of the errors is normal. PROC MIXED in SAS / MIXED SPSS used restricted maximum likelihood (REML) approach by default. REML equation by: Henderson,  1959; Laird et.al. 1982; Jennrich 1986; Lindstrom & Bates, 1988; Gurka et.al 2006.

#### REML

```math
logREML(\theta,\beta) = -\frac{N-p}{2} - \frac{1}{2}\sum_{i=1}^nlog|V_{i}|-

-\frac{1}{2}log|\sum_{i=1}^nX_i'V_i^{-1}X_i|-\frac{1}{2}\sum_{i=1}^n(y_i - X_{i}\beta)'V_i^{-1}(y_i - X_{i}\beta)
```

#### Beta (β)

```math
\beta = {(\sum_{i=1}^n X_{i}'V_i^{-1}X_{i})}^{-1}(\sum_{i=1}^n X_{i}'V_i^{-1}y_{i})
```

#### F

```math
F = \frac{\beta'L'(LCL')^{-1}L\beta}{rank(LCL')}
```

#### Variance covariance matrix of β

```math
C = (\sum_{i=1}^{n} X_i'V_i^{-1}X_i)^{-1}
```

#### Sweep

Details see: https://github.com/joshday/SweepOperator.jl
