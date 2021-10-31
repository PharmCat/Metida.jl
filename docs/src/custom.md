### Custom structures

*Custom structures have been remaken in version 0.12. Now custom structures more simple to construct and use.*

To make your own covariance structure you should make sruct that <: AbstractCovarianceType:

Example:

```
struct YourCovarianceStruct <: AbstractCovarianceType end
```

You also can specify additional field if you need to use them inside functions.

Then you can make function for construction random effect matrix (gmat!) and repeated effect (rmat!). Only upper triangular can be updated.

Function `gmat!` have 3 arguments: mx - zero matrix, θ - theta vector for this effect, and your custom structure object.

Next this function used to make random part of variance-covariance matrix: V' = Z * G * Z'

```
function Metida.gmat!(mx, θ, ::YourCovarianceStruct)
    @inbounds @simd for i = 1:size(mx, 1)
        mx[i, i] = θ[i] ^ 2
    end
    nothing
end
```

Function `rmat!` have 4 arguments and add repeated effect to V': V = V' + R (so V = Z * G * Z' + R), mx - V' matrix, θ - theta vector for this effect, rz - subject effect matrix, ct - your covariance type object. For example, `rmat!` for Heterogeneous Toeplitz Parameterized structure is specified bellow (`TOEPHP_  <: AbstractCovarianceType`).

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

One more function you shoud make is `covstrparam`, this function need to know how many parameters included in theta vector for optimization. Function returns number of variance parameters and rho parameters for this structure. Where t - number of columns in individual Z matrix for random effect number of columns in repeated effect matrix (rZ).

```
function Metida.covstrparam(ct::Union{ARH_, CSH_}, t::Int)::Tuple{Int, Int}
    return (t, 1)
end
```

For better printing you can add:

```
function Metida.rcoefnames(s, t, ct::ARMA_)
    return ["σ² ", "γ ", "ρ "]
end
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
