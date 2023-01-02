# Details

The solution to the mixed model equations is a maximum likelihood estimate when the distribution of the errors is normal. Maximum likelihood estimates are based on the probability model for the observed responses. In the probability model the distribution of the responses is expressed as a function of one or more parameters. PROC MIXED in SAS used restricted maximum likelihood (REML) approach by default. REML equation can be described with following (Henderson,  1959;Laird et.al. 1982; Jennrich 1986; Lindstrom & Bates, 1988; Gurka et.al 2006).

Metida.jl using optimization with Optim.jl package (Newton's Method) by default.  Because variance have only positive values and ρ is limited as -1 ≤ ρ ≤ 1 in Metida.jl "link" function is used. Exponential values is optimizing in variance part and ρ is linked with sigmoid function.
All steps perform with differentiable functions with forward automatic differentiation using ForwardDiff.jl package. Also [MetidaNLopt.jl](https://github.com/PharmCat/MetidaNLopt.jl) and [MetidaCu.jl](https://github.com/PharmCat/MetidaCu.jl) available for optimization with NLopt.jl and solving on CUDA GPU. Sweep algorithm using for variance-covariance matrix inversing in REML calculation.

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

#### REML

```math
logREML(\theta,\beta) = -\frac{N-p}{2} - \frac{1}{2}\sum_{i=1}^nlog|V_{\theta, i}|-

-\frac{1}{2}log|\sum_{i=1}^nX_i'V_{\theta, i}^{-1}X_i|-\frac{1}{2}\sum_{i=1}^n(y_i - X_{i}\beta)'V_{\theta, i}^{-1}(y_i - X_{i}\beta)
```

Actually ```L(\theta) = -2logREML = L_1(\theta) + L_2(\theta) + L_3(\theta) + c`` used for optimization, where:

```math
L_1(\theta) = \frac{1}{2}\sum_{i=1}^nlog|V_{i}| \\

L_2(\theta) = \frac{1}{2}log|\sum_{i=1}^nX_i'V_i^{-1}X_i| \\

L_3(\theta) = \frac{1}{2}\sum_{i=1}^n(y_i - X_{i}\beta)'V_i^{-1}(y_i - X_{i}\beta)
```

```math
\nabla\mathcal{L}(\theta) = \nabla L_1(\theta) + \nabla L_2(\theta) + \nabla L_3(\theta)
```

```math
\mathcal{H}\mathcal{L}(\theta) =  \mathcal{H}L_1(\theta)  + \mathcal{H}L_2(\theta) +  \mathcal{H} L_3(\theta)
```

##### Initial step

Initial (first) step before optimization may be done:

```math
\theta_{n+1} = \theta_{n} - \nabla\mathcal{L}(\theta_{n}) * \mathcal{H}^{'}(\theta_{n}) , where \\

\mathcal{H}^{'}(\theta_{n}) = - \mathcal{H}L_1(\theta_{n})  + \mathcal{H} L_3(\theta) , if score \\

\mathcal{H}^{'}(\theta_{n}) =  \mathcal{H} L_3(\theta_{n}) , if ai

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

#### [Variance parameters link function](@id varlink_header)

Apply special function to some part of theta vector.

##### Variance (var) part

Applied only to variance part.

###### Exponential function (:exp)

Exponential function applied.

```math
  f(x) = exp(x)
```

```math
  f^{-1}(x) = log(x)
```

###### Square function (:sq)

```math
  f(x) = x^2
```

```math
  f^{-1}(x) = sqrt(x)
```

###### Identity function (:identity)

```math
  f(x) = x
```

```math
  f^{-1}(x) = x
```

##### Covariance (rho) part

Applied only to covariance part.

###### Sigmoid function (:sigm)

```math
  f(x) = 1 / (1 + exp(- x * k)) * 2 - 1
```

```math
  f^{-1}(x) = -log(1 / (x + 1) * 2 - 1) / k
```

where ``k = 0.1``

###### Arctangent function (:atan)

```math
  f(x) = atan(x)/pi*2
```

```math
  f^{-1}(x) = tan(x*pi/2)
```

###### "Square" sigmoid function (:sqsigm)

```math
  f(x) = x / \sqrt{1 + (x)^2}
```

```math
  f^{-1}(x) = sign(x) * \sqrt{x^2/(1 - x^2)}
```

###### Positive sigmoid function (:psigm)

```math
  f(x) = 1/(1 + exp(-x / 2))
```

```math
  f^{-1}(x) = -log(1/x - 1) * 2
```

##### Additional parameters (theta) part

No function applied.
