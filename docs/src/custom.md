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

Function `rmat!` have 4 arguments and add repeated effect to V': V = V' + R (so V = Z * G * Z' + R), `mx` - V' matrix, `θ` - theta vector for this effect, `rz` - subject effect matrix, `ct` - your covariance type object. For example, `rmat!` for Heterogeneous Toeplitz Parameterized structure is specified bellow (`TOEPHP_  <: AbstractCovarianceType`).

```
function Metida.rmat!(mx, θ, rz, ct::TOEPHP_)
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

function Metida.rmat!(mx, θ, rz, ::CustomCovarianceStructure)
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