## Example 1 - Continuous and categorical predictors

```@example lmmexample
using Metida, CSV, DataFrames, CategoricalArrays, MixedModels;

rds = CSV.File(joinpath(dirname(pathof(Metida)), "..", "test", "csv",  "1fptime.csv"); types = [String, String, Float64, Float64]) |> DataFrame

rds2 = CSV.File(joinpath(dirname(pathof(Metida)), "..", "test", "csv",  "ftdf3.csv"); types = [String,  Float64, Float64, String, String, String, String, String, Float64]) |> DataFrame


devday = CSV.File(joinpath(dirname(pathof(Metida)), "..", "test", "csv",  "devday.csv"); types = [Float64, String, String, String ]) |> DataFrame

nothing; # hide
```

![](plot1.png)

Metida result:

```@example lmmexample
lmm = LMM(@formula(response ~1 + factor*time), rds;
random = VarEffect(@covstr(1 + time|subject&factor), CSH),
)
fit!(lmm)
```

MixedModels result:

```@example lmmexample
fm = @formula(response ~ 1 + factor*time + (1 + time|subject&factor))
mm = fit(MixedModel, fm, rds, REML=true)
println(mm) #hide
```

## Example 2 - Two random factors (Penicillin data)

Metida:

```@example lmmexample

df          = CSV.File(joinpath(dirname(pathof(Metida)), "..", "test", "csv", "Penicillin.csv"); types = [String, Float64, String, String]) |> DataFrame
df.diameter = float.(df.diameter)

lmm = LMM(@formula(diameter ~ 1), df;
random = [VarEffect(@covstr(1|plate), SI), VarEffect(@covstr(1|sample), SI)]
)
fit!(lmm)
```

MixedModels:

```@example lmmexample

fm2 = @formula(diameter ~ 1 + (1|plate) + (1|sample))
mm = fit(MixedModel, fm2, df, REML=true)
println(mm) #hide
```

## Example 3 - Repeated ARMA/AR/ARH

```@example lmmexample
rds = CSV.File(joinpath(dirname(pathof(Metida)), "..", "test", "csv",  "1freparma.csv"); types = [String, String, Float64, Float64]) |> DataFrame

nothing # hide
```

![](plot2.png)

ARMA:

```@example lmmexample
lmm = LMM(@formula(response ~ 1 + factor*time), rds;
random = VarEffect(@covstr(factor|subject&factor), DIAG),
repeated = VarEffect(@covstr(1|subject&factor), ARMA),
)
fit!(lmm)
```

AR:

```@example lmmexample
lmm = Metida.LMM(@formula(response ~ 1 + factor*time), rds;
random = VarEffect(@covstr(factor|subject&factor), DIAG),
repeated = VarEffect(@covstr(1|subject&factor), AR),
)
fit!(lmm)
```

ARH:

```@example lmmexample
lmm = Metida.LMM(@formula(response ~ 1 + factor*time), rds;
random = VarEffect(@covstr(factor|subject&factor), DIAG),
repeated = VarEffect(@covstr(1|subject&factor), ARH),
)
fit!(lmm)
```

## Example 4 - SAS relation

### Model 1

```
df0 = CSV.File(joinpath(dirname(pathof(Metida)), "..", "test", "csv", "df0.csv")) |> DataFrame

lmm = LMM(@formula(var ~ sequence + period + formulation), df0;
random   = VarEffect(@covstr(formulation|subject), CSH),
repeated = VarEffect(@covstr(formulation|subject), DIAG),
)
fit!(lmm)
```

SAS code:

```
PROC MIXED data=df0;
CLASSES subject sequence period formulation;
MODEL  var = sequence period formulation/ DDFM=SATTERTH s;
RANDOM  formulation/TYPE=CSH SUB=subject G V;
REPEATED/GRP=formulation SUB=subject R;
RUN;
```

### Model 2

```
lmm = LMM(
    @formula(var ~ sequence + period + formulation), df0;
    random   = VarEffect(@covstr(formulation|subject), SI),
    repeated = VarEffect(@covstr(formulation|subject), DIAG),
)
fit!(lmm)
```

SAS code:

```
PROC MIXED data=df0;
CLASSES subject sequence period formulation;
MODEL  var = sequence period formulation/ DDFM=SATTERTH s;
RANDOM  formulation/TYPE=VC SUB=subject G V;
REPEATED/GRP=formulation SUB=subject R;
RUN;
```

### Model 3

```
lmm = LMM(@formula(var ~ sequence + period + formulation), df0;
    random = VarEffect(@covstr(subject|1), SI)
    )
fit!(lmm)
```

SAS code:

```
PROC MIXED data=df0;
CLASSES subject sequence period formulation;
MODEL  var = sequence period formulation/ DDFM=SATTERTH s;
RANDOM  subject/TYPE=VC G V;
RUN;
```

## Example 5 - Working with Effects.jl

```
using Effects, StatsModels

lmm = LMM(@formula(var ~ sequence + period + formulation), df0;
    random = VarEffect(@covstr(subject|1), SI)
    )
fit!(lmm)

table_model = StatsModels.TableRegressionModel(lmm, lmm.mf, lmm.mm)

emmeans(tm)

effects(Dict(:period => ["1", "2", "3", "4"]), tm)
```


## Unstructured covariance 

Unstructured covariance example.


Metida result:

```@example lmmexample
lmm = Metida.LMM(@formula(response~factor), rds2;
    random = Metida.VarEffect(Metida.@covstr(r1|subject), UN),
    )
Metida.fit!(lmm)
```

MixedModels result:

```@example lmmexample
mm = fit(MixedModel, @formula(response ~ factor+ (0+r1|subject)), rds2, REML = true)
println(mm) #hide
```

## Covariance modifiers (Experimental)

Two structures in Metida are not covariance structures in their own right but
**modifiers of an R-part block that has already been built**: `ACOV(type)` adds a
correlation term to it, `BAND(k)` masks one out of it. Both are used as a
*subsequent* repeated effect and both read what the preceding effects wrote.

```
  repeated = [ VarEffect(@covstr(<variance factor> | <block 1>), <structure>),
               VarEffect(@covstr(1                 | <block 2>), <modifier>)  ]
```

A modifier in the first position is an error: at that point the block is still
all zeros, so there is nothing to augment or to mask and the modifier's
parameter is unidentifiable.

### `ACOV` - augmented covariance

`ACOV` reads the variances already sitting on the diagonal of the block and adds
a correlation computed from them:

```math
R_{mn} \;\mathrel{+}=\; \sqrt{R_{mm} R_{nn}} \cdot \varrho^{\,|n-m|}
\qquad \texttt{ACOV(AR)}
```

```math
R_{mn} \;\mathrel{+}=\; \sqrt{R_{mm} R_{nn}} \cdot \varrho
\qquad \texttt{ACOV(CS)}
```

```math
R_{mn} \;\mathrel{+}=\; \sqrt{R_{mm} R_{nn}} \cdot \varrho^{\,d(m,n)}
\qquad \texttt{ACOV(SPPOW)}
```

where ``m, n`` are row positions **inside the sub-block selected by the modifier's
own blocking factor**, and ``d(m,n)`` is the Euclidean distance between rows
``m`` and ``n`` of the modifier's model matrix. The modifier contributes exactly
one parameter, ``\varrho``, and no variance parameters.

The point of this is to **separate the variance model from the correlation
model**. A built-in structure ties the two together: `ARH(r1|subject)` means
"heterogeneous variances by `r1`, autoregressive correlation across the rows of
`subject`" - one factor, one blocking. With `ACOV` the two halves are written
independently, each with its own factor and its own blocking.

#### Equivalence identities

With the same blocking factor on both effects, the composition reproduces the
built-in structure. These are the self-checks:

| Composition | Equivalent built-in | ``-2\log REML`` |
|---|---|---|
| `SI` + `ACOV(AR)` | `AR` | 833.887 |
| `DIAG` + `ACOV(AR)` | `ARH` | 821.402 |
| `DIAG` + `ACOV(CS)` | `CSH` | 862.131 |

The two parameterisations reach points about ``10^{-12}`` apart - ordinary
optimiser tolerance, not different models.

```@example lmmexample
lmm1 = Metida.LMM(@formula(response ~ 1), rds2;
    repeated = [Metida.VarEffect(Metida.@covstr(r1|subject), Metida.DIAG),
                Metida.VarEffect(Metida.@covstr(1|subject), Metida.ACOV(Metida.AR))])
Metida.fit!(lmm1)
```

```@example lmmexample
lmm2 = Metida.LMM(@formula(response ~ 1), rds2;
    repeated = [Metida.VarEffect(Metida.@covstr(r1|subject), Metida.ARH)])
Metida.fit!(lmm2)
```

R-part of the variance-covariance matrix:

```@example lmmexample
Metida.rmatrix(lmm1, 1)
```

So on its own this composition buys nothing. It is the *decoupled* use - a
different factor or a different blocking on each half - that `ACOV` exists for.

#### The `action` keyword

`action` decides what happens when the cell the modifier is about to write into
is **already not zero**:

| `action` | behaviour on a nonzero cell |
|---|---|
| 0 (default) | add |
| 1 | replace |
| 2 | keep the existing value |
| 3 / 4 / 5 | as 0 / 1 / 2, with a warning at construction |
| > 5 | error |

On a **diagonal** base (`SI`, `DIAG`) no covariance cell is ever nonzero, so
every value of `action` yields the same model (821.402 for `DIAG` + `ACOV(AR)`),
and a warning says the keyword has no effect. On a non-diagonal base the values
differ as intended:

| base | `action` | ``-2\log REML`` |
|---|---|---|
| `CS(r1\|subject)` | 0 - add | 834.553 |
| `CS(r1\|subject)` | 1 - replace | 833.887 |
| `CS(r1\|subject)` | 2 - keep | 876.651 |

`action = 1` erases the `CS` covariance and leaves only the `AR` increment,
which is why it coincides with `SI` + `ACOV(AR)` and with plain `AR` (833.887).
`action = 2` makes ``\varrho`` inert on every cell the base already filled.

### `ACOV(SPPOW)` - lag as a distance, not a row number

!!! warning "`AR` lag is a row position, not a time value"
    `ACOV(AR)` - exactly like the built-in `AR`, `ARH` and `TOEP` - uses
    ``|n-m|``, the difference of **row positions inside the block**, as the lag.
    No time column is consulted, so reordering the rows of the data frame
    changes the model.

`ACOV(SPPOW)` takes the lag from the Euclidean distance between the rows of its
own model matrix, so the correlation follows the actual spacing of a time or
coordinate variable and does not depend on how the rows happen to be ordered:

| row order | `ACOV(AR)` | `ACOV(CS)` | `ACOV(SPPOW)` |
|---|---|---|---|
| as supplied | 821.402 | 862.131 | 821.402 |
| sorted by `r1` descending | 812.900 | 862.131 | 821.402 |
| sorted by `subject`, then by the time column | 821.402 | 862.131 | 821.402 |

`ACOV(CS)` is exchangeable and therefore immune by construction. `ACOV(SPPOW)`
is immune because it reads the coordinate. On an evenly spaced grid it
reproduces `ACOV(AR)` exactly.

The coordinate must be **numeric**: passed as a categorical column it becomes an
indicator matrix, every pair of distinct rows is then at distance ``\sqrt 2``,
and the structure silently degenerates into `ACOV(CS)`.

```@example lmmexample
rds2.pn = parse.(Float64, rds2.p)   # the time column as a number, not a factor

lmm = Metida.LMM(@formula(response ~ 1), rds2;
    repeated = [Metida.VarEffect(Metida.@covstr(r1|subject), Metida.DIAG),
                Metida.VarEffect(Metida.@covstr(pn|subject), Metida.ACOV(Metida.SPPOW))])
Metida.fit!(lmm)
```

For unequally spaced visits this is the only correct choice of the three.

### `BAND(k)` - banding an arbitrary structure

`BAND(k)` zeroes every covariance cell whose within-block lag exceeds `k`,
leaving the diagonal and the first `k` lags untouched. It contributes **no
parameters**:

```math
R_{mn} \;\leftarrow\; 0 \quad \text{for } |n-m| > k
```

Banded variants previously existed only for the Toeplitz family (`TOEPP`,
`TOEPHP`). `BAND` supplies them for `CSH`, `ARH`, `UN`, the spatial structures
and anything else that can precede it - covariance tapering with no extra
parameters.

```@example lmmexample
lmm = Metida.LMM(@formula(response ~ 1), rds2;
    repeated = [Metida.VarEffect(Metida.@covstr(r1|subject), Metida.CSH),
                Metida.VarEffect(Metida.@covstr(1|subject), Metida.BAND(2))])
Metida.fit!(lmm)
```

```@example lmmexample
Metida.rmatrix(lmm, 1)
```

Cross-check against an existing structure, and the no-op at the far end:

| model | ``\theta`` length | ``-2\log REML`` |
|---|---|---|
| `CSH` + `BAND(1)` | 4 | 963.058 |
| `TOEPHP(2)` | 4 | **963.058** |
| `CSH` + `BAND(6)` (``k \ge`` block size) | 4 | 862.131 |
| `CSH` | 4 | **862.131** |

!!! note "`BAND(k)` across `k` is a family of equal-dimension, non-nested models"
    Masking fixes no free parameter - the ``\theta`` of `CSH` + `BAND(k)` is the
    ``\theta`` of `CSH` for every `k`. So two different `k` give two
    4-parameter families, neither contained in the other, and ``-2\log REML``
    need not be monotone in `k`:

    | `k` | 1 | 2 | 3 | 4 | 5 | 6 |
    |---|---|---|---|---|---|---|
    | ``-2\log REML`` | 963.058 | 934.844 | 934.894 | 925.058 | 896.997 | 862.131 |

    The ``k=3`` value above is worse than ``k=2``, and it is reproducible from
    either starting point - this is the non-nesting, not a convergence failure.
    Compare candidates by AIC, and do not read a likelihood-ratio test into a
    change of `k`.

!!! note "Why there is no `BLOCK` modifier"
    A modifier that zeroed correlations according to a *factor* would be
    redundant. ``\log|V|`` and ``r'V^{-1}r`` are block separable, so a
    factor-shaped mask is numerically identical to blocking by that factor -
    same answer, computed more slowly on larger blocks padded with zeros. And
    that masking already comes for free: each effect writes only into the
    sub-blocks its own factor selects, so zeros between the levels of a factor
    appear by themselves.

    A **band** is different: it is not block diagonal. Lag 1 connects row 3 to
    row 4 across any grouping boundary, and no blocking factor expresses that.
    That is what `BAND` is for.

### Blocking - read this before choosing factors

!!! warning "A modifier's blocking factor enters the model's partition"
    Metida builds the block-diagonal partition of ``V`` from the subject
    groupings of all random effects and of every **non-diagonal** repeated
    effect. `ACOV` and `BAND` are both non-diagonal, so **their factors join
    that union**, while a `SI` or `DIAG` first effect does not.

    Consequence: in a model whose only non-diagonal effect is the modifier, the
    blocking of the whole model is dictated by the modifier alone. Narrowing its
    factor narrows every block, which is a **different model**, not a refinement
    of the same one:

    | `repeated` | blocks | block size | ``-2\log REML`` |
    |---|---|---|---|
    | `DIAG(r1\|subject)` + `ACOV(AR)(1\|subject)` | 24 | 7 | 821.402 |
    | `DIAG(r1\|subject)` + `ACOV(AR)(1\|subject&r2)` | 48 | 5 | 800.761 |
    | `DIAG(r1\|subject)` + `ACOV(AR)(1\|subject&r1)` | 72 | 3 | 798.992 |
    | `CS(r1\|subject)` + `ACOV(AR)(1\|subject&r1)` | 24 | 7 | 799.613 |

    To apply a correlation **only inside nested sub-blocks while keeping the
    parent block intact**, the first effect must itself be non-diagonal (the
    last row): `CS` keeps the subject-level block, and `ACOV` then adds
    correlation only within `subject & r1`.

```@example lmmexample
lmm = Metida.LMM(@formula(response ~ 1), rds2;
    repeated = [Metida.VarEffect(Metida.@covstr(r1|subject), Metida.CS),
                Metida.VarEffect(Metida.@covstr(1|subject&r1), Metida.ACOV(Metida.AR))])
Metida.fit!(lmm)
Metida.rmatrix(lmm, 1)
```

Rows 1–3 of that block belong to `r1 == "A"`, rows 4–5 to `"B"`, rows 6–7 to
`"C"`: the `CS` covariance is present everywhere, the `ACOV` increment only
within each `r1` group.

The same holds for `BAND`: `CSH(r1|subject)` + `BAND(1)(1|subject&r1)` keeps the
24 blocks of size 7 (because `subject & r1` is nested in `subject`, and the
union is `subject`) while banding only inside each `r1` group - 932.220 against
963.058 for the same band applied to the whole block.

### Nested / multi-level structures

This is where a modifier has no built-in equivalent. The `devday` data has four
devices measured on two days for each subject. The structure is built in two
layers: the first describes the unstructured device-device covariance within a
day, the second adds the day-day correlation within each device.

```@example lmmexample
lmm4 = Metida.LMM(@formula(resp ~ 0 + device), devday;
    repeated = [Metida.VarEffect(Metida.@covstr(device|subj&day), Metida.UN),
                Metida.VarEffect(Metida.@covstr(1|subj&device), Metida.ACOV(Metida.AR))])
Metida.fit!(lmm4)
```

```@example lmmexample
Metida.rmatrix(lmm4, 1)
```

One parameter buys a drop from 986.813 (`UN` alone, 10 parameters) to 903.058
(11 parameters).

!!! note "The result is additive, not separable"
    The composition above is **not** a Kronecker product ``UN \otimes AR``. Each
    layer fills only the cells its own blocking factor selects, so cells that
    differ in *both* device and day receive no contribution and stay exactly
    zero:

    | cell | value | source |
    |---|---|---|
    | same device, different day (`A d1`, `A d2`) | 13.551 | `ACOV(AR)` |
    | same day, different device (`A d1`, `B d1`) | 0.746 | `UN` |
    | different device **and** day (`A d1`, `B d2`) | 0 | neither |

    A separable structure would put ``\sigma_{ij}\varrho`` in the third row.
    Twelve of the 28 off-diagonal cell pairs of the 8×8 block are structural
    zeros. This is a legitimate and parsimonious model - one correlation
    parameter against the ``\binom{8}{2}`` of a full `UN` - but it must be
    reported as what it is.

### Application patterns

**1. Variance by one factor, correlation by another.** Variances differ by
analyte, assay or treatment while the correlation follows a different grouping
(period, visit, batch). Keep the first effect non-diagonal so the parent block
survives:

```julia
repeated = [VarEffect(@covstr(r1|subject),   CS),        # variance + base correlation
            VarEffect(@covstr(1|subject&r2), ACOV(CS))]  # extra correlation within r2
```

**2. Technical replicates nested in a sample.** Replicates of the same sample
are more alike than different samples: `DIAG`/`UN` across samples, `ACOV(CS)`
within the replicate group - one parameter for the intra-replicate correlation.

**3. Longitudinal measurement of several endpoints.** `UN` across endpoints
within a visit plus `ACOV(SPPOW)` across visits within an endpoint - the
`devday` pattern, with `SPPOW` rather than `AR` whenever the visits are unevenly
spaced.

**4. Tapering a rich structure.** `UN` or `CSH` with `BAND(k)` keeps the
heterogeneous variances and the near-diagonal covariances and discards the
far-lag ones, at no cost in parameters. Useful when the block is large relative
to the number of subjects and a full `UN` will not converge.

**5. Replicate bioequivalence designs.** Heterogeneous within-subject variances
by formulation (`DIAG(formulation|subject)`) with a period-driven correlation
added on top. Compare against `CSH`/`ARH`/`UN` by AIC before adopting it: on
balanced data the built-ins are usually enough, and a modifier earns its place
only when the variance and correlation groupings genuinely differ.

**6. Reproducing a structure from other software.** Because the two halves are
written separately, a composition can express some SAS `PROC MIXED` structures
that have no single Metida name. Always confirm the resulting `rmatrix(lmm, i)`
cell by cell against the reference implementation - the additive behaviour above
means an apparently equivalent specification can differ in its structural zeros.

### Pitfalls

**Boundary ``\hat\varrho``.** A modifier applied inside small sub-blocks easily
drives the correlation to the edge of the parameter space. The link functions map
``\mathbb{R}`` onto the *open* interval, so ``|\hat\varrho| = 1`` is never
reached exactly, but an estimate that close to it means the likelihood is still
rising at the boundary: ``V`` is numerically singular in that direction, the
Hessian is ill conditioned there, and Satterthwaite degrees of freedom for any
contrast loading on that parameter are not trustworthy. `fit!` logs a warning
when ``1 - |\hat\varrho| <`` `rhotol` (``10^{-8}`` by default):

```
Correlation parameter 3 (ρ) is on the boundary: ρ = 0.9999999999989,
1 - |ρ| < 1.0e-8. Variance-covariance matrix is nearly singular in this
direction; DF and SE can be unreliable.
```

The fourth row of the blocking table above is exactly such a model: it fits to
799.613 with ``\hat\theta = [6.414, -0.024, 1.000]`` and one warning in the log.
Check `getlog(lmm)` and `theta(lmm)`, and when ``\hat\varrho`` sits on the
boundary, simplify in that direction (`UN → CSH`, `CSH → CS`, `ARH → AR`), merge
factor levels, or re-cut the blocks. Set `rhotol = 0.0` to switch the check off.

**Positive definiteness** is not guaranteed by construction, for `ACOV` because
the increment can overwhelm the diagonal and for `BAND` because zeroing
covariances can break the ordering of eigenvalues. Non-positive-definite blocks
are caught by the Cholesky factorisation and reported through `getlog(lmm)`;
`lmm.result.ipd` carries the verdict.

**Optimiser sensitivity.** Masked and augmented structures can have flatter
likelihoods than the built-ins. The default Newton method is recommended; on
`CSH` + `BAND(3)`, for instance, `optmethod = LBFGS_OM` returns 6510.457 against
Newton's 934.894. Always look at `-2logREML` before interpreting a fit.

### Limitations

| | |
|---|---|
| Supported `ACOV` inner types | `AR`, `CS`, `SPPOW`. The others are rejected at construction with `Unknown covariance type!` |
| Position in `repeated` | a modifier must not be first - there is nothing to modify yet; this is an error, not a warning |
| `ACOV(SPPOW)` coordinate | must be numeric; a categorical column degenerates the structure into `ACOV(CS)` |
| `BAND(k)` | `k >= 0`; `k` at least the block size is a no-op; different `k` are not nested models |
| `action` | only has an effect on a cell that is already nonzero, i.e. only on a non-diagonal base |
| Boundary estimates | ``\hat\varrho`` can land on ``\pm 1``; a warning is logged, see `rhotol` |
| Positive definiteness | not guaranteed; check `lmm.result.ipd` and `getlog(lmm)` |


