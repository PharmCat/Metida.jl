# Denominator degrees of freedom

Inference on fixed effects in a mixed model requires a denominator degrees of
freedom (DDF) value, because the covariance parameters ``\theta`` are estimated
rather than known. Metida provides three methods, described below in the order of
increasing cost.

---

## 1. Notation

| Symbol | Meaning | Code |
|---|---|---|
| ``N,\ p,\ t`` | observations, ``\operatorname{rank}(X)``, number of covariance parameters | `nobs`, `rankx`, `thetalength` |
| ``\theta`` | covariance parameters at the optimum | `lmm.result.theta` |
| ``V(\theta)`` | marginal covariance ``ZGZ^{\prime}+R`` | — |
| ``C`` | ``\left(X^{\prime}V^{-1}X\right)^{-1}``, covariance of ``\hat\beta`` | `lmm.result.c`, `vcov(lmm)` |
| ``H`` | Hessian of the profiled ``-2\ell_R`` with respect to ``\theta`` | `lmm.result.h` |
| ``W`` | asymptotic covariance of ``\hat\theta``, ``W = 2H^{-1}`` | `getinvhes(lmm)` |
| ``l`` | contrast vector, length ``p`` | — |
| ``L`` | contrast matrix, ``k\times p`` | `lcontrast(lmm, i)` |
| ``g`` | vector with ``g_i = l^{\prime}(\partial C/\partial\theta_i)\,l`` | — |
| ``\nu_i,\ E`` | per-direction DF and their Fai–Cornelius sum | `vm`, `em` |


---

## 2. Residual degrees of freedom

```math
\mathrm{DDF} = N - p
```

Implemented as `dof_residual(lmm) = nobs(lmm) - lmm.rankx`. Exact only when
``V`` is known up to a single scale factor — that is, for a model whose covariance
structure reduces to ``\sigma^2 I``. For any other structure it is
anti-conservative, sometimes severely so: a between-subject effect in a repeated
measures design gets ``N-p`` instead of roughly the number of subjects.

Available as `ddf = :residual` in `confint`, `typeiii` and `contrast`. Useful as a
sanity bound and as a fallback, not as a default.

---

## 3. Containment

`dof_contain(lmm, i)` reproduces the containment rule of `PROC MIXED`. For
coefficient ``i``:

1. Find the model term ``T`` that coefficient ``i`` belongs to
   (`lmm.modstr.assign[i]`), and its set of variables
   ``\mathcal{V}(T)`` = `StatsModels.termvars`.
2. For every random effect ``r`` whose own variables intersect ``\mathcal{V}(T)``,
   compute
```math
\mathrm{rank}\!\left(\big[\,X \;\; Z_r\,\big]\right)
```
   where ``Z_r`` is the **full** design matrix of that random effect — the
   Kronecker expansion over all its subjects, of size ``N \times (q_r \cdot n_r)``.
3. Return the minimum over those ranks. If no random effect contains the term,
   return ``N - \mathrm{rank}([X\;Z])`` with ``Z`` the concatenation of all
   ``Z_r``.


Available as `ddf = :contain`. Returns an `Int`.

!!! note
    Containment DDF are integers determined by the design alone — they do not
    depend on ``\hat\theta``. That makes them reproducible but insensitive to the
    actual covariance structure.

---

## 4. Satterthwaite: the three building blocks

The Satterthwaite family needs three ingredients, each computed once per fitted
model and cached.

### 4.1 The Hessian ``H``

`reml_hessian(lmm)` returns the Hessian of the **profiled** criterion — the same
function the optimizer minimized, with ``\hat\beta(\theta)`` substituted
internally.

It is obtained by central finite differences of the **exact** automatic-differentiation
gradient:

```math
H_{\cdot i} \;=\; \frac{\nabla F(\theta + h_i e_i) - \nabla F(\theta - h_i e_i)}{2 h_i},
\qquad
h_i = \max\!\left(|\theta_i|,\,1\right)\cdot \varepsilon^{1/3}
```

followed by symmetrization ``H \leftarrow \tfrac{1}{2}(H + H^{\prime})``.


Why finite differences of an exact gradient rather than nested AD:

* the differentiated quantity ``\nabla F`` is exact, so only one order of
  discretization error enters. With a central difference and the step above the
  relative accuracy is ``\mathcal{O}(\varepsilon^{2/3}) \approx 4\cdot 10^{-11}``,
  far tighter than the ``10^{-2}`` tolerance at which DDF are typically reported;
* nested AD would require specializing the whole block pass for
  `Dual{Dual{Float64,c},c}`, whose element carries ``(1+c)^2`` floats. That path is
  never exercised by `fit!`, so it must be compiled from scratch — the dominant
  cost for a quantity evaluated once per fit;
* the cost is ``2t`` gradient evaluations, against roughly ``(1+c)`` gradient-equivalents
  for nested AD — comparable at run time, negligible at compile time.


### 4.2 The asymptotic covariance ``W``

`getinvhes(lmm)` returns ``W = 2H^{-1}``, computed through a **truncated spectral
inverse** rather than a direct inversion:

```math
H = Q\Lambda Q^{\prime}, \qquad
\mathcal{K} = \{\,i : \lambda_i > \tau\,\}, \qquad
\tau = \max_j |\lambda_j| \cdot \sqrt{\varepsilon}\, t
```

```math
W = 2\, Q_{\mathcal{K}}\,\Lambda_{\mathcal{K}}^{-1}\,Q_{\mathcal{K}}^{\prime}
```

Three properties follow from keeping only *positive* eigenvalues above the
threshold:

* ``W`` is positive semidefinite **by construction**, so ``d = g^{\prime}Wg \ge 0``
  always. A negative variance estimate for the contrast is impossible.
* Directions in which the criterion is flat (unidentified parameters, parameters
  driven to a boundary) are removed. Because the threshold is relative to
  ``\max|\lambda|``, the criterion is scale-free and does not change when ``N`` or
  the units of the response change.
* Near-collinearity between covariance parameters is caught. A test on
  ``|H_{ii}|`` alone would not: two nearly collinear parameters can both have large
  diagonal entries while the smallest eigenvalue is negligible.

Truncation is reported to the log. If ``H`` contains non-finite entries, or if no
eigenvalue survives, `getinvhes` returns a matrix of `NaN` and records an `:ERROR`
message — so the failure propagates to `NaN` DDF rather than silently producing a
number.

!!! note
    SAS and SPSS **remove individual parameters** that hit a boundary from the
    asymptotic covariance. Metida truncates **directions** in the joint parameter
    space. For an interior optimum with a positive definite ``H`` the two coincide
    exactly and both give ``2H^{-1}``. They can differ on models where an estimate
    sits on a boundary.

### 4.3 The gradient of ``C``

`gradc(lmm, theta)` returns ``\{\partial C/\partial\theta_i\}_{i=1}^{t}``. Since
``C = \theta_2^{-1}`` with ``\theta_2 = X^{\prime}V^{-1}X``, the derivative of an
inverse gives

```math
\frac{\partial C}{\partial \theta_i}
= -\,C \;\frac{\partial \theta_2}{\partial \theta_i}\; C
```

and ``\partial\theta_2/\partial\theta_i`` is obtained as a ForwardDiff Jacobian of
`sweep_β_cov`, which returns ``\theta_2(\theta)``. The ``\beta`` argument passed to
`sweep_β_cov` is inert: ``\theta_2`` does not depend on ``\beta``, and the residual
term it would affect is discarded.

The result is cached in `lmm.result.grc`. Evaluation is always at
`lmm.result.theta`.

---

## 5. Satterthwaite for a single contrast

For a contrast vector ``l``, the estimated variance of ``l^{\prime}\hat\beta`` is
``l^{\prime}Cl``, itself a random quantity because ``C = C(\hat\theta)``. A
first-order delta-method expansion gives

```math
\widehat{\operatorname{Var}}\!\left(l^{\prime}\hat{C}l\right) \approx g^{\prime}Wg,
\qquad g_i = l^{\prime}\frac{\partial C}{\partial \theta_i}l
```

Matching the first two moments of a scaled ``\chi^2`` yields

```math
\boxed{\;\mathrm{DDF} = \frac{2\left(l^{\prime}Cl\right)^{2}}{g^{\prime}Wg}\;}
```


### Entry points

| Call | Behaviour |
|---|---|
| `dof_satter(lmm, l::AbstractVector)` | contrast of length `coefn(lmm)`; reduced to `rankx` via `pivotvec` when ``X`` is rank deficient |
| `dof_satter(lmm, i::Int)` | unit contrast for coefficient ``i``; returns `NaN` if that coefficient was dropped by the rank reduction |
| `dof_satter(lmm)` | vector of length `coefn(lmm)`, one entry per coefficient, `NaN` in dropped positions |
| `dof_satter(lmm, L::AbstractMatrix)` | multi-row contrast, see §6 |

### Return values and guards

| Condition | Returned | Logged |
|---|---|---|
| ``\mathrm{DDF} \le 0`` | `NaN` | `:ERROR` |
| ``0 < \mathrm{DDF} < 1`` | ``1`` | — |
| ``\mathrm{DDF} > N-p`` | ``N-p`` | — |
| otherwise | ``\mathrm{DDF}`` | — |

The lower clamp keeps the resulting ``t`` distribution defined. The upper clamp is a
safeguard, not part of the method: Satterthwaite DDF have no analytic upper bound,
and a value above ``N-p`` indicates that the delta-method variance
``g^{\prime}Wg`` came out implausibly small.

---

## 6. Satterthwaite for a multi-row contrast (Fai–Cornelius)

For an ``L`` with ``k>1`` rows the statistic is an ``F``, and the DDF follow the
Fai–Cornelius construction. Let

```math
LCL^{\prime} = Q\Lambda Q^{\prime}, \qquad
q = \operatorname{rank}\!\left(LCL^{\prime}\right)
```

be the spectral decomposition, and let ``\ell_i`` be the ``i``-th row of
``Q^{\prime}L``. Each eigen-direction is treated as an independent single contrast:

```math
\nu_i = \frac{2\,\lambda_i^{2}}{g_i^{\prime} W g_i},
\qquad g_{i,m} = \ell_i^{\prime}\frac{\partial C}{\partial\theta_m}\ell_i
```

The identity that makes ``\lambda_i`` the right numerator is
``Q^{\prime}LCL^{\prime}Q = \Lambda``, so ``\ell_i^{\prime} C \ell_i = \lambda_i``
— exactly the ``l^{\prime}Cl`` of §5 for that direction.

Summing only the directions with more than two degrees of freedom,

```math
E = \sum_{i\,:\,\nu_i>2} \frac{\nu_i}{\nu_i - 2},
\qquad
\boxed{\;\mathrm{DDF} = \frac{2E}{E - q}\;}
```

Three implementation details matter:

**Explicit symmetrization.** ``LCL^{\prime}`` is symmetric mathematically but is
stored as a general `Matrix`, so rounding makes it asymmetric at the level of the
last bits. Calling `eigen` on such a matrix dispatches to the general LAPACK path
(`geevx!`, with balancing), whose eigenvectors are not guaranteed orthonormal and
whose behaviour on nearly degenerate spectra varies between BLAS versions.
Wrapping in `Symmetric` selects `syevr!`: real eigenvalues, orthonormal
eigenvectors — which is what makes ``Q^{\prime} = Q^{-1}`` valid in the formula
above.

**Selecting the largest eigenvalues.** `eigen(Symmetric(·))` returns eigenvalues in
ascending order. When ``LCL^{\prime}`` is rank deficient, the leading entries are
the numerically zero ones; taking `1:lclr` would use exactly the directions that
carry no information. The range is therefore taken from the top.

**Rows of ``Q^{\prime}L``, not of ``L``.** `plm = view(pl, k, :)` avoids allocating
a row per direction inside the loop.

### Return values

Same guard table as §5. The important case is ``E \le q``: the denominator
``E-q`` is then non-positive, the DDF comes out ``\le 0``, and `NaN` is returned.
This is not a numerical accident — it means that no direction (or almost none)
achieved ``\nu_i > 2``, so the Fai–Cornelius approximation does not apply to this
contrast under this fit.

!!! note "Reachable range"
    For a given ``q``, the map ``E \mapsto 2E/(E-q)`` is monotonically decreasing on
    ``E > q``, from ``+\infty`` down to the asymptote ``2``. Values in ``(1, 2]`` are
    therefore **unreachable**, and a returned DDF of exactly ``1`` for ``q \ge 1``
    always means the lower clamp fired.

---

## 7. Where DDF are used

| Function | Default | Accepted |
|---|---|---|
| `confint(lmm; ddf = ...)` | `:satter` | `:satter`, `:contain`, `:residual` |
| `typeiii(lmm; ddf = ...)` | `:satter` | `:satter`, `:contain`, `:residual` |
| `contrast(lmm, L; ddf = ...)` | `:satter` | `:satter`, `:residual`, or a number |
| `estimate(lmm, l)` | `:satter` | — |
| `coeftable(lmm)` | — | uses a ``z`` statistic, **no** DDF |

!!! warning
    `coeftable`, and therefore the table printed by `show(lmm)`, uses a normal
    approximation rather than a ``t`` distribution with Satterthwaite DDF. On small
    samples this understates ``p``-values. Use `estimate`, `confint` or `typeiii`
    for inference.

Single-coefficient hypotheses use the ``t`` distribution with the DDF from §5;
multi-row hypotheses use ``F(k, \mathrm{DDF})`` with the DDF from §6 and

```math
F = \frac{\left(L\hat\beta\right)^{\prime}\left(LCL^{\prime}\right)^{-}\left(L\hat\beta\right)}
         {\operatorname{rank}(L)}
```

(`fvalue`, using a pseudo-inverse of ``LCL^{\prime}``).

---

## 8. Choosing a method

**`:satter`** is the default and the right choice in almost all cases: it adapts to
the covariance structure and to the imbalance of the design.

**`:contain`** is worth using when reproducing a published `PROC MIXED` analysis
that specified `DDFM=CONTAIN`, or when ``\hat\theta`` sits on a boundary and the
Satterthwaite machinery returns `NaN`. It ignores ``\hat\theta`` entirely, which is
both its robustness and its weakness.

**`:residual`** is for balanced designs with a single variance component, and as an
upper reference point.

Cost: `:residual` is free; `:satter` costs one Hessian (``2t`` gradient
evaluations) plus one Jacobian, both cached; `:contain` costs one SVD of an
``N \times (p + q_r n_r)`` matrix per random effect, also cached in the vectorized
call.

---

## 9. Limitations

**The Fai–Cornelius statistic is not invariant to the choice of eigenbasis.** When
``LCL^{\prime}`` has a repeated eigenvalue, the basis of the corresponding
eigenspace is arbitrary, and ``E = \sum \nu_i/(\nu_i-2)`` is a nonlinear function
of the per-direction variances ``g_i^{\prime}Wg_i``. Rotating within a degenerate
eigenspace therefore changes the DDF. This is a property of the method, not of the
implementation; balanced designs with multi-level factors routinely produce
degenerate spectra. Agreement with other software on such a model depends on both
implementations happening to select a compatible basis.

**Truncation is a discrete decision.** A covariance parameter whose eigenvalue sits
near ``\tau`` is either fully retained or fully removed. If ``\hat\theta``
approaches a boundary, small changes in the fit — including changes in floating
point summation order caused by a different thread count — can flip that decision
and move the DDF. Fix `maxthreads` for reproducible reporting, and check
`getlog(lmm)` for truncation warnings.

**Rank-deficient ``X``.** Contrasts are reduced through `lmm.pivotvec`. The
vectorized `dof_satter(lmm)` and the vector method handle this; contrasts built by
hand must have `coefn(lmm)` columns, matching what `lcontrast` produces.

**Delta method is first order.** All Satterthwaite variants approximate the
sampling distribution of ``l^{\prime}\hat Cl`` by a scaled ``\chi^2`` matched on two
moments, using a first-order expansion in ``\hat\theta - \theta``. The
approximation degrades when the number of subjects is small relative to the number
of covariance parameters. Kenward–Roger, which additionally corrects ``C`` itself
for the estimation of ``\theta``, is the standard remedy and is not currently
implemented.

---
