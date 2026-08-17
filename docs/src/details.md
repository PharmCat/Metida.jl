# Details

The solution to the mixed model equations is a maximum likelihood estimate when the distribution of the errors is normal. Maximum likelihood estimates are based on the probability model for the observed responses. In the probability model the distribution of the responses is expressed as a function of one or more parameters. PROC MIXED in SAS used restricted maximum likelihood (REML) approach by default. REML equation can be described with following (Henderson,  1959;Laird et.al. 1982; Jennrich 1986; Lindstrom & Bates, 1988; Gurka et.al 2006).

Metida.jl using optimization with Optim.jl package (Newton's Method) by default.  Because variance have only positive values and ρ is limited as -1 ≤ ρ ≤ 1 in Metida.jl "link" function is used. Exponential values is optimizing in variance part and ρ is linked with sigmoid function.
All steps perform with differentiable functions with forward automatic differentiation using ForwardDiff.jl package. Also [MetidaNLopt.jl](https://github.com/PharmCat/MetidaNLopt.jl) and [MetidaCu.jl](https://github.com/PharmCat/MetidaCu.jl) available for optimization with NLopt.jl and solving on CUDA GPU. Sweep algorithm using for variance-covariance matrix inversing in REML calculation.

## 1. Model and notation

The linear mixed model:

```math
y = X\beta + Zu + \varepsilon, \qquad
u \sim \mathcal{N}(0, G), \qquad
\varepsilon \sim \mathcal{N}(0, R), \qquad
u \perp \varepsilon
```

where:


* ``y \in \mathbb{R}^{N}`` is the response vector (`lmm.data.yv`);
* ``X \in \mathbb{R}^{N \times p}`` is the fixed-effects design matrix (`lmm.data.xv`),
  with ``p = \operatorname{rank}(X)`` (`lmm.rankx`);
* ``Z`` is the random-effects design matrix (`lmm.covstr.z`);
* ``G = G(\theta)`` is the covariance of the random effects (G-side);
* ``R = R(\theta)`` is the residual covariance (R-side);
* ``\theta \in \mathbb{R}^{t}`` is the vector of covariance parameters,
  ``t`` = `lmm.covstr.tl`.

The marginal covariance of the response:

```math
V(\theta) = Z G(\theta) Z^{\prime} + R(\theta)
```

If ``X`` is not of full column rank, columns are selected by pivoted QR; from that
point on the reduced ``X`` given by `lmm.pivotvec` is used everywhere, and
``p = \operatorname{rank}(X)``.

The unknown parameters include the regression parameters in ``\beta`` and covariance parameters in ``\theta``.

Estimation of these model parameters relies on the use of a Newton-Ralphson (by default) algorithm. When we use either algorithm for finding REML solutions, we need to compute ``V^{-1}`` and its derivatives with respect to ``\theta``, which are computationally difficult for large ``n``, therefor SWEEP (see https://github.com/joshday/SweepOperator.jl) algorithm used to meke oprtimization less computationaly expensive.

### Notation summary

Every symbol used in this document, with the section where it is introduced.

| Symbol | Meaning | §|
|---|---|---|
| ``y,\ X,\ Z,\ \beta,\ u,\ \varepsilon`` | response, fixed design, random design, fixed effects, random effects, errors | 1 |
| ``G,\ R,\ V`` | random-effect, residual and marginal covariance | 1 |
| ``N,\ p,\ t`` | observations, ``\operatorname{rank}(X)``, number of covariance parameters | 1 |
| ``\theta`` | covariance parameter vector; ``\sigma`` its `:var` entries, ``\rho`` its `:rho` entries | 1, 5 |
| ``\theta_1,\ \theta_2,\ \theta_3,\ c`` | the four terms of ``-2\ell_R`` | 2 |
| ``r,\ \hat\beta,\ \beta_m`` | residual, GLS estimate, ``X^{\prime}V^{-1}y`` | 2, 3 |
| ``C`` | accumulator ``[X\;y]^{\prime}V^{-1}[X\;y]`` of order ``p+1`` | 3 |
| ``n,\ B_i,\ q_i`` | number of blocks, block ``i``, its size | 4 |
| ``V_i,\ X_i,\ y_i`` | restrictions of ``V``, ``X``, ``y`` to block ``i`` | 4 |
| ``\eta,\ f`` | unconstrained parameters, link transform ``\theta=f(\eta)`` | 5 |
| ``G_r,\ q_r,\ \tau(i,j)`` | covariance of random effect ``r``, its size, `UN` index map | 6 |
| ``\xi_m,\ d_{mn},\ \theta_e`` | spatial coordinate row, distance, range parameter | 7 |
| ``w,\ D_i,\ M`` | weight vector, ``\operatorname{diag}(w^{-1/2})``, weight matrix | 8 |
| ``s`` | positions of one subject inside a block | 9 |
| ``U_i,\ U_2`` | upper Cholesky factors of ``V_i`` and of ``\theta_2`` | 10, 13 |
| ``A_i,\ \tilde A_i`` | augmented block ``[X_i\;y_i]`` and its whitened form ``U_i^{-\prime}A_i`` | 10 |
| ``n_c`` | number of threads | 12 |
| ``b`` | half-solve ``U_2^{-\prime}\beta_m`` | 13 |
| ``F,\ H`` | objective ``-2\ell_R\circ f``, its Hessian | 14 |
| ``L`` | contrast matrix (inference layer only) | 15 |

---

## 2. The REML criterion

Metida minimizes ``-2\ell_R``:

```math
-2\ell_R(\theta) \;=\; \underbrace{\log|V|}_{\theta_1}
\;+\; \underbrace{\log\left|X^{\prime}V^{-1}X\right|}_{\log\det\theta_2}
\;+\; \underbrace{r^{\prime}V^{-1}r}_{\theta_3}
\;+\; \underbrace{(N-p)\log 2\pi}_{c}
```

where the residual is taken at the generalized least squares estimate

```math
\hat\beta(\theta) = \left(X^{\prime}V^{-1}X\right)^{-1} X^{\prime}V^{-1}y,
\qquad r = y - X\hat\beta(\theta)
```

The names ``\theta_1, \theta_2, \theta_3`` match the variable names in the source
(`reml.jl`). The constant is computed as `c = (N - lmm.rankx)*log(2π)`.

Quantities exposed to the user:

```math
\texttt{m2logreml} = -2\ell_R, \qquad
\texttt{logreml} = \ell_R = -\tfrac{1}{2}\,(-2\ell_R)
```

---

## 3. What actually has to be computed

The criterion looks as though it needs ``V^{-1}``. It does not. Expanding the
quadratic form at ``\hat\beta``:

```math
r^{\prime}V^{-1}r = y^{\prime}V^{-1}y - 2\hat\beta^{\prime}X^{\prime}V^{-1}y
+ \hat\beta^{\prime}X^{\prime}V^{-1}X\hat\beta
```

and since ``\hat\beta`` solves ``\theta_2\hat\beta = \beta_m`` with
``\theta_2 = X^{\prime}V^{-1}X`` and ``\beta_m = X^{\prime}V^{-1}y``, the last two
terms collapse:

```math
\boxed{\;\theta_3 = y^{\prime}V^{-1}y - \hat\beta^{\prime}\beta_m\;}
```

So the entire criterion is determined by **four scalars/small matrices**:

```math
\log|V|, \qquad X^{\prime}V^{-1}X, \qquad X^{\prime}V^{-1}y, \qquad y^{\prime}V^{-1}y
```

The last three are the blocks of a single symmetric matrix of order ``p+1``:

```math
C \;=\; [X \;\; y]^{\prime} V^{-1} [X \;\; y] \;=\;
\begin{pmatrix}
X^{\prime}V^{-1}X & X^{\prime}V^{-1}y \\[2pt]
y^{\prime}V^{-1}X & y^{\prime}V^{-1}y
\end{pmatrix}
```

Everything below is organised around accumulating ``\log|V|`` and ``C``.

---

## 4. Block partitioning

``V`` is block diagonal. Observations are split into independent blocks
``B_1,\dots,B_n`` (`lmm.covstr.vcovblock`) such that any two observations in
different blocks have zero covariance:

```math
V = \operatorname{blockdiag}(V_1, \dots, V_n)
```

The blocks are the connected components of the union of all subject groupings
induced by the random effects and by the non-diagonal repeated effects. Diagonal
structures (``\mathrm{SI}``, ``\mathrm{DIAG}``) do not link observations and take
no part in the union.

Both accumulated quantities are therefore plain sums over blocks:

```math
\log|V| = \sum_{i=1}^{n}\log|V_i|, \qquad
C = \sum_{i=1}^{n} [X_i \;\; y_i]^{\prime} V_i^{-1} [X_i \;\; y_i]
```

The full matrix ``V`` is **never formed**: work proceeds one block at a time, each
of size ``q_i = |B_i|``. The largest block size is `lmm.maxvcbl`.

*Source:* `varstruct.jl`, the `CovStructure` constructor; fields `vcovblock`, `esb`.

---

## 5. Parameterization of ``\theta`` and link functions

Every element of ``\theta`` carries a type (`lmm.covstr.ct`):

| Type | Meaning | Domain |
|---|---|---|
| `:var` | **standard deviation** ``\sigma`` | ``\sigma > 0`` |
| `:rho` | correlation ``\rho`` | ``\rho \in (-1,1)`` or ``(0,1)`` |
| `:theta` | other (spatial range, etc.) | ``\mathbb{R}`` |

!!! note
    For `:var` the vector ``\theta`` stores **``\sigma``, not ``\sigma^2``** — the
    covariance builders square it. This matters when cross-checking against
    SAS/SPSS, which report variances, and whenever converting between scales.

Optimization is carried out in an unconstrained space ``\eta \in \mathbb{R}^{t}``.
The forward transform ``\theta = f(\eta)`` is applied elementwise
(`varlinkvecapply`); the inverse ``\eta = f^{-1}(\theta)`` is used when preparing
the starting point (`varlinkrvecapply!`).

**Variance parameters** (`varlinkf`):

```math
\texttt{:exp:}\quad \sigma = e^{\eta}, \qquad \eta = \log\sigma
```
```math
\texttt{:sq:}\quad \sigma = \eta^{2}, \qquad \eta = \sqrt{\sigma}
```
```math
\texttt{:identity:}\quad \sigma = \eta
```

For `:exp`, when ``\eta < -21`` the constant ``7.5825604\cdot10^{-10}`` is returned
instead (guard against underflow).

**Correlation parameters** (`rholinkf`):

```math
\texttt{:sigm}\ (\text{default}):\quad
\rho = \frac{2}{1+e^{-\eta/10}} - 1 \in (-1,1)
```
```math
\texttt{:atan:}\quad \rho = \frac{2}{\pi}\arctan\eta \in (-1,1)
```
```math
\texttt{:sqsigm:}\quad \rho = \frac{\eta}{\sqrt{1+\eta^{2}}} \in (-1,1)
```
```math
\texttt{:psigm:}\quad \rho = \frac{1}{1+e^{-\eta/2}} \in (0,1)
```

The inverse transforms:

```math
\eta = -10\log\!\left(\frac{2}{\rho+1}-1\right),\quad
\eta = \tan\frac{\pi\rho}{2},\quad
\eta = \operatorname{sign}(\rho)\sqrt{\frac{\rho^{2}}{1-\rho^{2}}},\quad
\eta = -2\log\!\left(\frac{1}{\rho}-1\right)
```

Parameters of type `:theta` are not transformed.

The choice of link function changes the geometry of the optimization problem and
may shift the located optimum slightly — which is why `m2logreml` agrees across
different `rholinkf` settings only to within convergence tolerance.

*Source:* `utils.jl`, functions `vlink`, `vlinksq`, `rholinksigmoid`,
`rholinksigmoidatan`, `rholinksqsigmoid`, `rholinkpsigmoid`, and the wrappers
`varlinkvecapply` / `varlinkrvecapply!`.

---


## 6. Building ``G``

The vector ``\theta`` is cut into disjoint ranges (`lmm.covstr.tr`): one per random
effect, then one per repeated effect.

For random effect ``r`` a matrix ``G_r`` of size ``q_r \times q_r`` is built, where
``q_r`` is the number of columns of ``Z`` belonging to that effect. Below ``\theta``
denotes the slice belonging to the effect in question and ``q`` is the block size.

**ScaledIdentity (`SI`)**, 1 parameter:
```math
G_{ii} = \theta_1^{2}, \qquad G_{ij} = 0
```

**Diagonal (`DIAG`)**, ``q`` parameters:
```math
G_{ii} = \theta_i^{2}, \qquad G_{ij} = 0
```

**CompoundSymmetry (`CS`)**, 2 parameters:
```math
G_{ii} = \theta_1^{2}, \qquad G_{ij} = \theta_1^{2}\theta_2
```

**HeterogeneousCompoundSymmetry (`CSH`)**, ``q+1`` parameters:
```math
G_{ii} = \theta_i^{2}, \qquad G_{ij} = \theta_i\theta_j\theta_{q+1}
```

**Autoregressive (`AR`)**, 2 parameters:
```math
G_{ii} = \theta_1^{2}, \qquad G_{ij} = \theta_1^{2}\,\theta_2^{|i-j|}
```

**HeterogeneousAutoregressive (`ARH`)**, ``q+1`` parameters:
```math
G_{ii} = \theta_i^{2}, \qquad G_{ij} = \theta_i\theta_j\,\theta_{q+1}^{|i-j|}
```

**ARMA (`ARMA`)**, 3 parameters:
```math
G_{ii} = \theta_1^{2}, \qquad
G_{ij} = \theta_1^{2}\,\theta_2\,\theta_3^{|i-j|-1}
```

**Toeplitz (`TOEP`)**, ``q`` parameters:
```math
G_{ii} = \theta_1^{2}, \qquad G_{ij} = \theta_1^{2}\,\theta_{|i-j|+1}
```

**ToeplitzParameterized (`TOEPP(k)`)**, ``k`` parameters — the same, but banded:
```math
G_{ij} = \begin{cases}
\theta_1^{2}\,\theta_{|i-j|+1}, & 0 < |i-j| < k\\
0, & |i-j| \ge k
\end{cases}
```

**HeterogeneousToeplitz (`TOEPH`)**, ``2q-1`` parameters:
```math
G_{ii} = \theta_i^{2}, \qquad G_{ij} = \theta_i\theta_j\,\theta_{q+|i-j|}
```

**HeterogeneousToeplitzParameterized (`TOEPHP(k)`)** — the banded variant of `TOEPH`.

**Unstructured (`UN`)**, ``q + q(q-1)/2`` parameters:
```math
G_{ii} = \theta_i^{2}, \qquad
G_{ij} = \theta_i\theta_j\,\theta_{q + \tau(i,j)},\quad i<j
```
where the correlation index is numbered row-wise over the upper triangle:
```math
\tau(i,j) = \sum_{k=1}^{i-1}(q-k) \; + \; (j-i)
```

For the placeholder effect `RZero` (a model with no random part) ``G`` is not built.

*Source:* `gmat.jl`, functions `gmatvec` (builds the vector ``\{G_r\}``) and `gmat!`
(fills the upper triangle for a particular structure). Storage is `Symmetric`; only
the upper triangle is used.

---


## 7. Building ``R``

``R`` is filled block by block and **additively**: when several repeated effects are
specified, their contributions are summed:

```math
R_i = \sum_{j=1}^{m} R_i^{(j)}
```

Within block ``i``, for repeated effect ``j`` the subject sub-blocks ``s`` are
iterated (the positions of one subject's observations inside the block), and the
matrix of the corresponding structure is added into ``R_i[s,s]``. The formulas for
``\mathrm{SI}, \mathrm{DIAG}, \mathrm{CS}, \dots, \mathrm{UN}`` are identical to
section 6.

The R-side additionally offers spatial structures. Let
``d_{mn} = \lVert \xi_m - \xi_n \rVert_2`` be the Euclidean distance between rows of the
coordinate matrix `rz`, and ``\sigma^2 = \theta_1^2``:

**SpatialExponential (`SPEXP`)**:
```math
R_{mn} = \sigma^{2}\exp\!\left(-\frac{d_{mn}}{\theta_e}\right),
\qquad \theta_e = |\theta_2|
```

**SpatialPower (`SPPOW`)**:
```math
R_{mn} = \sigma^{2}\rho^{\,d_{mn}}, \qquad \rho = \theta_2
```

**SpatialGaussian (`SPGAU`)**:
```math
R_{mn} = \sigma^{2}\exp\!\left(-\frac{d_{mn}^{2}}{\theta_e^{2}}\right),
\qquad \theta_e = \theta_2
```

In all three, ``R_{mm} = \sigma^{2}``.

!!! warning
    For ``\mathrm{SPEXP}`` and ``\mathrm{SPGAU}``, when ``|\theta_2| < \varepsilon``
    the value ``\sqrt{\varepsilon}`` is substituted; this introduces a discontinuity
    in the objective near zero. For ``\mathrm{SPPOW}`` with ``\rho<0`` and
    non-integer ``d_{mn}``, the expression ``\rho^{d}`` is undefined in real
    arithmetic.

*Source:* `rmat.jl`, `rmat_base_inc!` and the `rmat!` family.

---


## 8. Observation weights

When weights are supplied they are applied to the **R side**, before ``ZGZ'`` is
added.

**Weight vector** ``w``: for block ``i`` the matrix
``D_i = \operatorname{diag}\big(w_{i1}^{-1/2},\dots,w_{iq}^{-1/2}\big)`` is formed,
and

```math
R_i \;\leftarrow\; D_i\, R_i\, D_i, \qquad\text{that is}\qquad
(R_i)_{mn} \;\leftarrow\; \frac{(R_i)_{mn}}{\sqrt{w_m w_n}}
```

A larger weight corresponds to a smaller residual variance. A constant weight scales
``R`` as a whole and therefore does not change ``-2\ell_R``: the scale is absorbed
by the estimate of ``\sigma^2``.

**Weight matrix** ``M``: the submatrix ``M_i = M[B_i, B_i]`` is used, and an
elementwise (Hadamard) product is applied:

```math
R_i \;\leftarrow\; R_i \circ M_i
```

*Source:* `lmmdata.jl` (`LMMWts`, which stores ``w^{-1/2}``), `utils.jl`
(`applywts!`), `linearalgebra.jl` (`mulβdαβd!`).

---


## 9. Assembling ``V_i``

For block ``i`` the order of operations is strictly as follows:

1. ``V_i \leftarrow 0``
2. ``V_i \leftarrow V_i + R_i`` — `rmat_base_inc!`
3. apply weights — `applywts!`
4. ``V_i \leftarrow V_i + Z_i G Z_i^{\prime}`` — `zgz_base_inc!`

Step 4 proceeds over subject sub-blocks: for each random effect ``r`` and each
subject occupying positions ``s`` within the block,

```math
V_i[s,s] \;\leftarrow\; V_i[s,s] + Z_i[s,\cdot]\,G_r\,Z_i[s,\cdot]^{\prime}
```

The kernel ``\Theta \leftarrow \Theta + ABA^{\prime}`` is hand-written and fills
**only the upper triangle** (`mulαβαtinc!`). Consequently the whole matrix ``V_i``
is held as an upper triangle in memory, the lower triangle stays zero, and
everything downstream uses the `Symmetric(·, :U)` wrapper.

*Source:* `utils.jl`, `vmatrix!(V, G, θ, rθ, lmm, i)` — the entry point used inside
REML; the public counterpart is `vmatrix(lmm, i)`.

---


## 10. The block pass: Cholesky and triangular solves

This is the computational core, implemented once in `_reml_blocks` and shared by
every consumer of the criterion — the objective value, the gradient, the Hessian,
and the alternative optimizer back-ends.

For each block:

**Step 1 — Cholesky factorization of ``V_i``.**

```math
V_i = U_i^{\prime} U_i, \qquad U_i \ \text{upper triangular}
```

!!! note "Notation"
    ``U_i`` denotes the Cholesky factor of the block. It has nothing to do with the
    residual covariance ``R`` of §7, nor with the random-effect vector ``u`` of §1.
    The factor is upper triangular because the whole package stores symmetric
    matrices as upper triangles (§9) and calls `potrf!('U', ...)`.

Performed in place on the upper triangle of ``V_i``:
`cholesky!(Symmetric(V, :U), check = false)`. Failure is detected through the
returned `info`/`issuccess` — a non-positive-definite ``V_i`` immediately yields
``+\infty`` for the objective.

**Step 2 — log determinant, for free.**

```math
\log|V_i| = 2\sum_{k=1}^{q_i}\log (U_i)_{kk}
```

**Step 3 — one triangular solve against the augmented right-hand side.**

Write ``A_i = [X_i \;\; y_i] \in \mathbb{R}^{q_i\times(p+1)}`` for the augmented
block, and ``\tilde A_i`` for its whitened counterpart:

```math
\tilde A_i = U_i^{-\prime} A_i
```

**Step 4 — one symmetric rank-``(p+1)`` update.**

```math
C \;\leftarrow\; C + \tilde A_i^{\prime}\tilde A_i
```

The identity that makes this work:

```math
\tilde A_i^{\prime}\tilde A_i = A_i^{\prime} U_i^{-1}U_i^{-\prime} A_i
= A_i^{\prime}\left(U_i^{\prime}U_i\right)^{-1} A_i
= A_i^{\prime}V_i^{-1}A_i
```

so ``V_i^{-1}`` is obtained implicitly and never materialized.


*Source:* `reml.jl`, `_reml_blocks`; kernels `_ltsolve!` (triangular solve) and
`_syrk_upper!` (symmetric update).

!!! note "Historical note"
    Versions through <= 0.17.2 used the sweep operator on the augmented matrix
    ``\begin{pmatrix} V_i & X_i \\ X_i^{\prime} & 0\end{pmatrix}``, whose
    ``1{:}q_i`` sweep produces
    ``\begin{pmatrix} -V_i^{-1} & V_i^{-1}X_i \\ X_i^{\prime}V_i^{-1} &
    -X_i^{\prime}V_i^{-1}X_i\end{pmatrix}``.
    That scheme also gave ``-V_i^{-1}`` explicitly.
    The sweep implementation remains in `sweep.jl` but is
    no longer on the REML path.

---


## 11. Dispatch: one implementation, two arithmetic paths

`_reml_blocks` is generic in the element type ``T``. Every kernel it calls has two
methods:

| Operation | ``T <: BlasFloat`` | ``T = \texttt{ForwardDiff.Dual}`` |
|---|---|---|
| Cholesky of ``V_i`` | `LAPACK.potrf!` (blocked) | generic `_chol!` |
| ``U^{-\prime}A`` | `BLAS.trsm!` | generic `ldiv!` on `UpperTriangular` |
| ``C \mathrel{+}= \tilde A^{\prime}\tilde A`` | `BLAS.syrk!` | `_syrk_upper!` loop |
| ``U_2^{-\prime}\beta_m``, ``U_2^{-1}b`` | `BLAS.trsv!` | generic `ldiv!` |

Dispatch is automatic: `ForwardDiff.Dual{T,V,N} <: Real`, so `Symmetric{Dual}`
satisfies `RealHermSymComplexHerm` and `cholesky!` selects the generic path.

The consequence matters statistically, not just for performance: **the objective
value and its derivatives are produced by the same code**.

*Source:* `reml.jl`, kernels `_syrk_upper!`, `_ltsolve!`, `_usolve!`,
`_qform_upper`.

---

## 12. Accumulation over blocks and parallelism

Blocks are distributed across threads statically: with ``n`` blocks and ``n_c``
threads each thread receives a contiguous range of block indices, where
``n_c = \min(\text{num\_cores},\, n,\, \texttt{maxthreads})``. Each thread keeps its
own scratch buffers (one allocation per call rather than one per block) and its own
accumulators, which are then summed:

```math
\theta_1 = \sum_{t=1}^{n_c}\texttt{accld}[t], \qquad
C = \sum_{t=1}^{n_c}\texttt{accC}[t]
```

!!! note
    The summation order depends on ``n_c``, so ``-2\ell_R`` may differ in the last
    digits on machines with different core counts.

---


## 13. Final assembly

`_reml_blocks` returns ``(\theta_1, C, \texttt{noerror})``. The rest operates on
matrices of order ``p+1`` and costs ``\mathcal{O}(p^3)`` once per evaluation.

**1. Extract ``\theta_2``.** The leading ``p\times p`` block of ``C``:

```math
\theta_2 = X^{\prime}V^{-1}X, \qquad
\beta_m = C[1{:}p,\, p{+}1], \qquad
y^{\prime}V^{-1}y = C[p{+}1,\, p{+}1]
```

``\theta_2`` is copied before factorization, because it is also the returned
``iC`` from which `vcov(lmm)` is formed.

**2. Second Cholesky factorization.**

```math
\theta_2 = U_2^{\prime}U_2, \qquad
\log\left|X^{\prime}V^{-1}X\right| = 2\sum_{k=1}^{p}\log (U_2)_{kk}
```

**3. Half-solve.**

```math
b = U_2^{-\prime}\beta_m
```

**4. Residual quadratic form.** By the identity of §3, and since
``\hat\beta^{\prime}\beta_m = \beta_m^{\prime}\theta_2^{-1}\beta_m = b^{\prime}b``:

```math
\theta_3 = y^{\prime}V^{-1}y - b^{\prime}b
```

This is precisely the Schur complement of ``\theta_2`` in ``C``; equivalently, if
the whole ``(p{+}1)\times(p{+}1)`` matrix ``C`` were factorized, ``\theta_3`` would
be the square of its last diagonal element. Building ``\theta_3`` from ``b`` rather
than from ``\hat\beta^{\prime}\beta_m`` avoids one extra round of cancellation.

**5. Fixed effects.**

```math
\hat\beta = U_2^{-1}b = \theta_2^{-1}\beta_m
```

**6. Result.**

```math
-2\ell_R = \theta_1 + \log\left|X^{\prime}V^{-1}X\right| + \theta_3 + (N-p)\log 2\pi
```

The tuple returned by `reml_sweep_β` is ``(-2\ell_R,\ \hat\beta,\
X^{\prime}V^{-1}X,\ \theta_3,\ \texttt{noerror})``.

The covariance of the fixed-effect estimates:

```math
\widehat{\operatorname{Var}}(\hat\beta) = \left(X^{\prime}V^{-1}X\right)^{-1}
```

— this is `vcov(lmm)`, `lmm.result.c`; `stderror(lmm)` is the square root of its
diagonal.


---

## 14. Optimization

The objective as a function of the unconstrained parameters:

```math
F(\eta) = -2\ell_R\big(f(\eta)\big)
```

Starting values: ``\hat\sigma^2_{OLS}`` from the QR decomposition of ``X``, divided
by ``(\text{number of random effects} + 1)``, for variance parameters; zeros for
correlation parameters; then ``f^{-1}`` is applied.

The default method is Newton's (`Optim.Newton` with Hager–Zhang line search). The
gradient and Hessian are obtained by automatic differentiation (`ForwardDiff`)
through the same implementation of ``F`` — the generic path of §11. The chunk size
is ``\min(8, t)``.

After convergence:

```math
\hat\theta = f(\hat\eta)
```

and a final call to `reml_sweep_β` produces ``\hat\beta``,
``X^{\prime}V^{-1}X`` and the validity flag.

*Source:* `fit.jl` (`fit!`, `optstep!`), `utils.jl` (`initvar`, `hessian`).

---


## 15. Matrix decompositions used

A complete inventory, since several of them are easy to overlook.

**On the REML path (once per objective evaluation):**

| # | Decomposition | Where | Size | Frequency |
|---|---|---|---|---|
| 1 | Cholesky ``V_i = U_i^{\prime}U_i`` | `_reml_blocks`, step 1 | ``q_i\times q_i`` | once per block |
| 2 | Triangular solve ``U_i^{-\prime}A_i`` | `_ltsolve!` | ``q_i\times(p{+}1)`` | once per block |
| 3 | Symmetric rank-``(p{+}1)`` update | `_syrk_upper!` | ``(p{+}1)^2`` | once per block |
| 4 | Cholesky ``\theta_2 = U_2^{\prime}U_2`` | §13 step 2 | ``p\times p`` | once per evaluation |
| 5 | Triangular solves ``U_2^{-\prime}\beta_m``, ``U_2^{-1}b`` | `_ltsolve!`, `_usolve!` | ``p`` | once per evaluation |

No explicit matrix inverse is computed anywhere on this path.

**Outside the objective, once per model:**

| # | Decomposition | Where | Purpose |
|---|---|---|---|
| 6 | Pivoted QR of ``X`` | `LMM` constructor | rank detection, `pivotvec` |
| 7 | QR of ``X`` | `initvar` | OLS starting variance |

**In the inference layer, on demand:**

| # | Decomposition | Where | Purpose |
|---|---|---|---|
| 8 | Symmetric eigendecomposition of ``LCL^{\prime}`` | `dof_satter_` (matrix form) | Satterthwaite for multi-df contrasts |
| 9 | Cholesky / pseudo-inverse of ``H`` | `getinvhes` | asymptotic covariance ``2H^{-1}`` of ``\hat\theta`` |
| 10 | SVD (via `rank`, `pinv`) | `fvalue`, `typeiii`, `lcontrast` | contrast rank and generalized inverse of ``LCL^{\prime}`` |
| 11 | Inverse of ``X^{\prime}V^{-1}X`` | `vcov` | reported covariance of ``\hat\beta`` |
| 12 | Cholesky of ``V_i`` | `rand`, `bootstrap` | sampling from ``\mathcal{N}(X\beta, V_i)`` |

---

## 16. Summary diagram

```
θ (unconstrained η)
  │  varlinkvecapply             §5
  ▼
θ (constrained: σ, ρ)
  │  gmatvec → gmat!             §6
  ▼
{G_r}
  │
  _reml_blocks: for each block i = 1..n            §4, §10
  │   V_i ← 0
  │   rmat_base_inc! → rmat!     §7      V_i += R_i
  │   applywts!                  §8      V_i ← D V_i D
  │   zgz_base_inc! → mulαβαtinc! §9     V_i += Z G Z'
  │   cholesky!(Symmetric(V_i))          V_i = U'U
  │       ├─ 2Σ log U_kk  → accld
  │   _ltsolve!                          Ã_i = U^{-T}[X_i y_i]
  │   _syrk_upper!                       C += Ã_i' Ã_i
  ▼
θ₁ = Σ log|V_i|,   C = [X y]'V⁻¹[X y]            §12
  │
  │  θ₂ = C[1:p,1:p],  βm = C[1:p,p+1],  y'V⁻¹y = C[p+1,p+1]
  │  cholesky!(Symmetric(θ₂))  → θ₂ = U₂'U₂       §13
  │      ├─ log|θ₂| = 2Σ log (U₂)_kk
  │  b  = U₂^{-T} βm            → θ₃ = y'V⁻¹y − b'b
  │  β̂  = U₂^{-1} b
  ▼
-2ℓ_R = θ₁ + log|θ₂| + θ₃ + (N-p)log2π
```

---


## 17. Numerical caveats

**Failure detection is explicit.** A non-positive-definite ``V_i`` or ``\theta_2`` is
reported through the `info` value of the Cholesky factorization; the objective then
returns ``+\infty`` and `noerror` is `false`. There is no silent regularization of
the diagonal and no correction term added to the log determinant.

**``\theta_3 \le 0``.** The Schur complement is positive whenever ``C`` is positive
definite. A non-positive value signals a degenerate fit (zero residual variance, or
accumulated round-off) and sets `noerror` to `false`.

**Cancellation.** ``\theta_3 = y^{\prime}V^{-1}y - b^{\prime}b`` is a difference of
two positive quantities. Forming it through ``b`` rather than through
``\hat\beta^{\prime}\beta_m`` keeps both terms on the same scale, but for models
with an extremely good fit the relative accuracy of ``\theta_3`` is still limited by
the conditioning of ``C``.

**Dependence on thread count.** See the note in §12.

**Upper-triangle convention.** ``V_i``, ``G``, ``R`` and ``C`` are all maintained as
upper triangles only; the lower triangles hold zeros or stale values. Any code
reading these matrices must go through `Symmetric(·, :U)`.

The validity of a fit can be checked with `getlog(lmm)`: messages at level `:ERROR`
and `:WARN` record whenever the conditions above were triggered.
