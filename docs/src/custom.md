### Custom structures


To make your own covariance structure first you should make struct that <: AbstractCovarianceType:

Example:

```
struct YourCovarianceStruct <: AbstractCovarianceType end
```

You also can specify additional field if you need to use them inside `gmat!`/`rmat!` functions.

Then you can make function for construction random effect matrix (`gmat!`) and repeated effect (`rmat!`). Only upper triangular can be updated.

Function `gmat!` have 3 arguments: `mx` - zero matrix, `θ` - theta vector for this effect, and your custom structure object.

Next this function used to make "random" part of variance-covariance matrix: V' = Z * G * Z'

```
function Metida.gmat!(mx, θ, ::YourCovarianceStruct)
    @inbounds @simd for i = 1:size(mx, 1)
        mx[i, i] = θ[i] ^ 2
    end
    nothing
end
```

Function `rmat!` have 5 arguments and add repeated effect to V': V = V' + R (so V = Z * G * Z' + R), `mx` - V' matrix, `θ` - theta vector for this effect, `rz` - subject effect matrix, `ct` - your covariance type object, `sb` = block number. For example, `rmat!` for Heterogeneous Toeplitz Parameterized structure is specified bellow (`TOEPHP_  <: AbstractCovarianceType`).

```
function Metida.rmat!(mx, θ, rz, ct::TOEPHP_, ::Int)
    l     = size(rz, 2)
    vec   = rz * (θ[1:l])
    s   = size(mx, 1)
    if s > 1 && ct.p > 1
        for m = 1:s - 1
            for n = m + 1:(m + ct.p - 1 > s ? s : m + ct.p - 1)
                @inbounds  mx[m, n] += vec[m] * vec[n] * θ[n - m + l]
            end
        end
    end
    @inbounds @simd for m = 1:s
        mx[m, m] += vec[m] * vec[m]
    end
    nothing
end
```

One more function you shoud make is `covstrparam`, this function need to know how many parameters included in theta vector for optimization. Function returns number of variance parameters and rho parameters for this structure. Where `t` - number of columns in individual `Z` matrix for random effect or number of columns in repeated effect matrix (`rZ`).

Example for Heterogeneous Autoregressive and Heterogeneous Compound Symmetry structures:

```
function Metida.covstrparam(ct::Union{ARH_, CSH_}, t::Int)::Tuple{Int, Int}
    return (t, 1)
end
```

For better printing you can add:

```
function Metida.rcoefnames(s, t, ct::YourCovarianceStruct)
    return ["σ² ", "γ ", "ρ "]
end
```

Where, `s` - effect schema, `t` - number of parameters, this function returns names for your covariance structure for printing in LMM output.

Add this method for better printing:

```
function Base.show(io::IO, ct::YourCovarianceStruct)
    print(io, "YourCovarianceStruct")
end
```

Then just make model and fit it:

```
lmm = Metida.LMM(@formula(response ~ 1 + factor*time), ftdf2;
  random = Metida.VarEffect(Metida.@covstr(factor|subject&factor), YourCovarianceStruct()),
  repeated = Metida.VarEffect(Metida.@covstr(1|subject&factor), YourCovarianceStruct()),
  )
  Metida.fit!(lmm)
```

Example:

```@example lmmexample
using Metida, DataFrames, CSV, CategoricalArrays

spatdf       = CSV.File(joinpath(dirname(pathof(Metida)), "..", "test", "csv", "spatialdata.csv"); types = [Int, Int, String, Float64, Float64, Float64, Float64, Float64]) |> DataFrame
ftdf = CSV.File(joinpath(dirname(pathof(Metida)), "..", "test", "csv",  "1fptime.csv"); types = [String, String, Float64, Float64]) |> DataFrame
df0 = CSV.File(joinpath(dirname(pathof(Metida)), "..", "test", "csv",  "df0.csv"); types = [String, String, String, String,Float64, Float64, Float64]) |> DataFrame

struct CustomCovarianceStructure <: Metida.AbstractCovarianceType end
function Metida.covstrparam(ct::CustomCovarianceStructure, t::Int)::Tuple{Int, Int}
    return (t, 1)
end
function Metida.gmat!(mx, θ, ct::CustomCovarianceStructure)
    s = size(mx, 1)
    @inbounds @simd for m = 1:s
        mx[m, m] = θ[m]
    end
    if s > 1
        for m = 1:s - 1
            @inbounds @simd for n = m + 1:s
                mx[m, n] = mx[m, m] * mx[n, n] * θ[end]
            end
        end
    end
    @inbounds @simd for m = 1:s
        mx[m, m] = mx[m, m] * mx[m, m]
    end
    nothing
end

lmm = Metida.LMM(@formula(response ~1 + factor*time), ftdf;
random = Metida.VarEffect(Metida.@covstr(1 + time|subject&factor), CustomCovarianceStructure()),
)
Metida.fit!(lmm)

# for R matrix

function Metida.rmat!(mx, θ, rz, ::CustomCovarianceStructure, ::Int)
    vec = Metida.tmul_unsafe(rz, θ)
    rn    = size(mx, 1)
    if rn > 1
        for m = 1:rn - 1
            @inbounds @simd for n = m + 1:rn
                mx[m, n] += vec[m] * vec[n] * θ[end]
            end
        end
    end
        @inbounds  for m ∈ axes(mx, 1)
        mx[m, m] += vec[m] * vec[m]
    end
    nothing
end

lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
repeated = Metida.VarEffect(Metida.@covstr(period|subject), CustomCovarianceStructure()),
)
Metida.fit!(lmm)
```

### Custom distance estimation for spatial structures

If you want to use coordinates or some other structures for distance estimation you can define method [`Metida.edistance`](@ref) to calculate distance:

```@example lmmexample
function Metida.edistance(mx::AbstractMatrix{<:CartesianIndex}, i::Int, j::Int)
    return sqrt((mx[i, 1][1] - mx[j, 1][1])^2 + (mx[i, 1][2] - mx[j, 1][2])^2)
end
```

For example this method returns distance between two vectors represented as `CartesianIndex`.

Make vector of `CartesianIndex`:

```@example lmmexample
spatdf.ci = map(x -> CartesianIndex(x[:x], x[:y]), eachrow(spatdf))
```

Then use new column as "raw" variable with  [`Metida.RawCoding`](@ref) contrast and fit the model:

```@example lmmexample
lmm = Metida.LMM(@formula(r2 ~ f), spatdf;
    repeated = Metida.VarEffect(Metida.@covstr(ci|1), Metida.SPEXP; coding = Dict(:ci => Metida.RawCoding())),
    )
Metida.fit!(lmm)
```

### Example: circular autoregressive structure

Some factors are cyclic: day of week, hour of day, month, angular sector. The last level of such a factor is the neighbour of the first one, so the covariance between levels 1 and 2 should be equal to the covariance between levels 1 and ``s`` (``s`` - number of levels). `AR` gives ``\rho^{s-1}`` for the last pair instead.

The stationary correlation of the closed AR(1) process ``e_t = \rho e_{t-1} + u_t``, ``e_0 \equiv e_s``, is:

```math
r_k = \frac{\rho^k + \rho^{s-k}}{1 + \rho^s}, \qquad k = |i - j|
```

``r_k = r_{s-k}``, so levels 1 and ``s`` are neighbours. The matrix is positive definite for any ``s`` and ``|\rho| < 1``. ``\rho`` is the parameter of the process, not the correlation of neighbouring levels: ``r_1 = (\rho + \rho^{s-1}) / (1 + \rho^s)``.

The structure has two parameters, ``\sigma`` and ``\rho``, as `AR`. `gmat!` fills the upper triangle, one superdiagonal at a time:

```@example lmmexample
struct CircularAR <: Metida.AbstractCovarianceType end

function Metida.covstrparam(ct::CircularAR, t::Int)::Tuple{Int, Int}
    return (1, 1)
end

function Metida.gmat!(mx, θ, ::CircularAR)
    de = θ[1] ^ 2
    s  = size(mx, 1)
    @inbounds @simd for i = 1:s
        mx[i, i] = de
    end
    if s > 1
        ρ   = θ[2]
        den = de / (1 + ρ ^ s)
        for k = 1:s - 1                     # k-th superdiagonal: |i - j| = k
            v = den * (ρ ^ k + ρ ^ (s - k))
            @inbounds @simd for m = 1:s - k
                mx[m, m + k] = v
            end
        end
    end
    return mx
end

function Metida.rcoefnames(s, t, ct::CircularAR)
    return ["σ² ", "ρ "]
end

function Base.show(io::IO, ct::CircularAR)
    print(io, "CircularAR")
end
nothing # hide
```

Simulated data: each subject has its own profile over a 7-day cycle (random effect with circular correlation, ``\rho = 0.6``), observed during 3 cycles:

```@example lmmexample
using Random, LinearAlgebra

rng = MersenneTwister(1)
s, ns, nc = 7, 60, 3                         # levels (days), subjects, cycles per subject
ρ   = 0.6
R   = [(ρ^abs(i - j) + ρ^(s - abs(i - j))) / (1 + ρ^s) for i in 1:s, j in 1:s]
L   = cholesky(Symmetric(R)).L
cdf = DataFrame(subject = String[], day = Int[], response = Float64[])
for i in 1:ns
    b = L * randn(rng, s)                    # subject profile over the cycle
    for c in 1:nc, d in 1:s
        push!(cdf, ("S$i", d, 10 + sin(2π * (d - 1) / s) + b[d] + 0.5 * randn(rng)))
    end
end
cdf.day = categorical(cdf.day)               # levels 1, 2, ..., 7 in cycle order
nothing # hide
```

Fit the model with circular random effect:

```@example lmmexample
lmm = Metida.LMM(@formula(response ~ day), cdf;
    random = Metida.VarEffect(Metida.@covstr(day|subject), CircularAR()),
    )
Metida.fit!(lmm)
```

Estimated G matrix: the first row is symmetric, ``G_{1,2} = G_{1,7}``:

```@example lmmexample
round.(Metida.gmatrix(lmm, 1), digits = 3)
```

`AR` has the same number of parameters, so AIC compares the two structures directly:

```@example lmmexample
lmm_ar = Metida.LMM(@formula(response ~ day), cdf;
    random = Metida.VarEffect(Metida.@covstr(day|subject), Metida.AR),
    )
Metida.fit!(lmm_ar)
Metida.aic(lmm), Metida.aic(lmm_ar)
```

Notes:

* Choose between `AR` and `CircularAR` by the data, not by default. If the data have no wrap-around, `CircularAR` underestimates the variance of contrasts between the first and the last levels; if they have, `AR` overestimates it. A free `TOEP` fit gives the same hint: its last bands grow back.
* Columns of the random effect matrix follow the order of factor levels, so the levels must be in cycle order. String labels `"1"`, ..., `"12"` are sorted as `"1"`, `"10"`, `"11"`, `"12"`, `"2"`, ... and break the cycle; use a categorical variable with explicit levels: `categorical(x; levels = ...)`.
* Heterogeneous version is made in the same way as `ARH` from `AR`: keep ``\sigma_i`` on the diagonal, multiply ``r_k`` by ``\sigma_i \sigma_j`` and return `(t, 1)` from `covstrparam`.
* Only `gmat!` is defined: the structure is meant for a random effect of the cycle phase. Serial correlation of residuals in time is not circular (the last day of a week is followed by the first day of the next week), use a repeated structure on the whole time axis there. For simultaneous measurements arranged on a circle (angular sectors) no custom code is needed: put the levels on a circle and use a spatial structure, the Euclidean (chord) distance depends only on the circular lag:

```
df.cx = cos.(2π .* df.sector ./ s)
df.cy = sin.(2π .* df.sector ./ s)
repeated = Metida.VarEffect(Metida.@covstr(cx + cy|subject), Metida.SPEXP)
```
