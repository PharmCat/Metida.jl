################################################################################
#                      Type III estimable functions
#
# The Type III L for a term is the term's own coefficients averaged with equal
# weights over the levels of the CLASS variables that appear only in the terms
# containing it (Goodnight 1980). The weights are the column means of each
# such variable's contrast matrix, so the hypothesis does not depend on the
# cell frequencies.
#
# Containment: F contains E when F and E have the same continuous part and the
# CLASS variables of E are a proper subset of those of F.
################################################################################

# Layout of one term: per component, the CLASS variable (or `nothing`), its
# width and the column means of its contrast matrix; plus the names of the
# continuous components. Components keep the order StatsModels lays the columns
# out in, first component varying fastest. `nothing` if the term has a
# multi-column component that is not categorical, where that layout does not
# hold.

struct ContrastTable
    name::Vector{String}
    f::Vector{Float64}
    ndf::Vector{Float64}
    df::Vector{Float64}
    pval::Vector{Float64}
end

function termlayout(t, ncol::Int)
    comps = isa(t, InteractionTerm) ? collect(t.terms) : Any[t]
    syms  = Union{Symbol, Nothing}[]
    wid   = Int[]
    cmean = Vector{Float64}[]
    cls   = Set{Symbol}()
    cont  = Set{String}()
    for c in comps
        w = StatsModels.width(c)
        w == 0 && continue                                  # InterceptTerm{false}
        if isa(c, CategoricalTerm)
            m = c.contrasts.matrix
            push!(syms, c.sym); push!(cls, c.sym)
            push!(cmean, [sum(view(m, :, j)) / size(m, 1) for j in axes(m, 2)])
        else
            w == 1 || return nothing
            push!(syms, nothing)
            push!(cmean, [1.0])
            isa(c, InterceptTerm) || push!(cont, string(c))
        end
        push!(wid, w)
    end
    prod(wid) == ncol || return nothing
    return (syms = syms, wid = wid, cmean = cmean, cls = cls, cont = cont)
end

# F contains E
t3contains(F, E) = F.cont == E.cont &&
                   length(E.cls) < length(F.cls) && issubset(E.cls, F.cls)

# Block of L on the columns of a containing term F, one row per column of E
function t3block(F, E)
    stE, stF = termstrides(E.wid), termstrides(F.wid)
    A  = zeros(prod(E.wid), prod(F.wid))
    jE = ones(Int, length(E.wid))
    for ci in CartesianIndices(Tuple(F.wid))
        w = 1.0
        fill!(jE, 1)
        for k in eachindex(F.wid)
            s = F.syms[k]
            s === nothing && continue               # continuous: same factor in E
            kE = findfirst(==(s), E.syms)
            kE === nothing ? (w *= F.cmean[k][ci[k]]) : (jE[kE] = ci[k])
        end
        iszero(w) || (A[termcolumn(jE, stE), termcolumn(ci, stF)] = w)
    end
    return A
end

function lcontrast3(lmm::LMM, i::Int)
    trms = lmm.f.rhs.terms
    asg  = assign(lmm)
    ci   = findall(==(i), asg)
    isempty(ci) && return zeros(0, size(lmm.data.xv, 2))
    E = termlayout(trms[i], length(ci))
    isnothing(E) && return lcontrast_terms(lmm, i)
    L = zeros(length(ci), size(lmm.data.xv, 2))
    for (r, c) in enumerate(ci)
        L[r, c] = 1.0
    end
    for k in eachindex(trms)
        k == i && continue
        ck = findall(==(k), asg)
        isempty(ck) && continue
        F = termlayout(trms[k], length(ck))
        isnothing(F) && return lcontrast_terms(lmm, i)
        t3contains(F, E) && (L[:, ck] .= t3block(F, E))
    end
    return L
end
"""
    typeiii(lmm::LMM; ddf::Symbol = :satter, contrasts::Symbol = :terms)

One Wald F test per term of the fixed-effect formula.

Returns a table with, for each term, the numerator degrees of freedom `ndf`,
the F statistic `f`, the denominator degrees of freedom `df` and the p-value
`pval`.

`ddf` picks the denominator degrees of freedom: `:satter` (default,
[`dof_satter`](@ref)), `:contain` ([`dof_contain`](@ref)) or `:residual`.

`contrasts` picks the null hypothesis tested for each term:

  * `:terms` (default) - the coefficients of the term are zero in the
    parameterization the model was fitted in. For a factor coded as
    differences from its first level this is equality of its levels taken at
    the reference level of every other factor, so the hypothesis is
    conditional on those reference levels. For a single-column term it is the
    hypothesis [`coeftable`](@ref) tests for that coefficient.
  * `:typeiii` - the equally weighted marginal means defined by the term are
    equal (Searle, Speed & Milliken, 1980). `L` is the Type III estimable
    function of the term (Goodnight, 1980): the coefficients of the term
    averaged with equal weights over the levels of the CLASS variables that
    appear only in the terms containing it, so the hypothesis is free of the
    cell frequencies and of the choice of reference level.

The two hypotheses coincide when no other term contains the term being tested.
They differ for the intercept, for a main effect that also appears inside an
interaction, and for a model fitted without an intercept.

!!! note
    `contrasts = :typeiii` will become the default in the next breaking
    release.

Use [`contrast`](@ref) to test a hypothesis of your own, and
[`lcontrast`](@ref) to see the `L` used for a term.

# Example

```julia
lmm = Metida.LMM(@formula(response ~ 1 + factor * time), ftdf;
    random = Metida.VarEffect(Metida.@covstr(factor|subject&factor), Metida.DIAG))
Metida.fit!(lmm)
Metida.typeiii(lmm)
Metida.typeiii(lmm; contrasts = :typeiii)
Metida.typeiii(lmm; ddf = :residual)
```

# References

Goodnight, J. H. (1980) Tests of hypotheses in fixed effects linear models.
*Communications in Statistics - Theory and Methods* 9, 167–180.

Searle, S. R., Speed, F. M., Milliken, G. A. (1980) Population marginal means
in the linear model: an alternative to least squares means. *The American
Statistician* 34, 216–221.
"""
function typeiii(lmm::LMM; ddf::Symbol = :satter, contrasts::Symbol = :terms)
    if !isfitted(lmm) error("Model not fitted!") end
    contrasts in (:terms, :typeiii) ||
        throw(ArgumentError("unknown contrasts type, allowed: :terms, :typeiii (got $contrasts)"))
    c           = length(lmm.f.rhs.terms)
    d           = Vector{Int}(undef, 0)
    fac         = Vector{String}(undef, c)
    F           = Vector{Float64}(undef,c)
    df          = Vector{Float64}(undef, c)
    ndf         = Vector{Float64}(undef, c)
    pval        = Vector{Float64}(undef, c)
    for i = 1:c
        iterm = lmm.f.rhs.terms[i]
        
        if typeof(iterm) <: InterceptTerm{false}
            push!(d, i)
            fac[i] = ""
            continue
        end
        
        fac[i]  = tname(iterm)
        L       = lcontrast(lmm, i; contrasts = contrasts)
        F[i]    = fvalue(lmm, L)
        ndf[i]  = rank(L)
        if ddf == :satter
            df[i]  = dof_satter(lmm, L)
        elseif ddf == :contain
            df[i]  = dof_contain_f(lmm, i)
        elseif ddf == :residual
            df[i]  = dof_residual(lmm)
        else
            throw(ArgumentError("unknown ddf type, allowed: :satter, :contain, :residual (got $ddf)"))
        end
        if ndf[i] > 0 && df[i] > 0
            pval[i] = ccdf(FDist(ndf[i], df[i]), F[i])
        else
            pval[i] = NaN
        end
    end
    if length(d) > 0
        deleteat!(fac, d)
        deleteat!(F, d)
        deleteat!(ndf, d)
        deleteat!(df, d)
        deleteat!(pval, d)
    end
    ContrastTable(fac, F, ndf, df, pval)
end

"""
    contrast(lmm, l::AbstractMatrix; name::String = "Contrast", ddf = :satter)

Wald F test of `L β = 0` for a contrast matrix `l` you supply, with
`size(l, 2) == coefn(lmm)`.

`ddf` = `:satter` ([`dof_satter`](@ref)) or `:residual`, or any number to set
the denominator degrees of freedom directly.

Use it for a hypothesis that is not one of the terms of the formula;
[`typeiii`](@ref) covers the terms, and [`lcontrast`](@ref) returns the `L` it
uses for each of them.

# Example

```julia
# resp ~ 0 + device: are the four device means all zero?
contrast(lmm, Matrix{Float64}(I, 4, 4); name = "device = 0")

# is the mean of A and B equal to the mean of C and D?
contrast(lmm, [0.5 0.5 -0.5 -0.5]; name = "AB vs CD")
```
"""
function contrast(lmm, l::AbstractMatrix; name::String = "Contrast", ddf = :satter)
    if !isfitted(lmm) error("Model not fitted!") end
    if coefn(lmm) != size(l, 2) error("size(l, 2) = $(size(l, 2)) not equal number of fixed effect coefficients = $(coefn(lmm))!") end
    F    = fvalue(lmm, l)
    ndf  = rank(l)
    if isa(ddf, Symbol)
        if ddf == :satter
            df  = dof_satter(lmm, l)
        elseif ddf == :residual
            df  = dof_residual(lmm)
        end
    else
        df = ddf
    end
    if ndf > 0 && df > 0
        pval = ccdf(FDist(ndf, df), F)
    else
        pval = NaN
    end
    return ContrastTable([name], [F], [ndf], [df], [pval])
end

function Base.show(io::IO, at::ContrastTable)
    mx = metida_table(at.name,  at.f, at.ndf, at.df, at.pval; names = (:Name, :F, :ndf, :ddf, :pval))
    PrettyTables.pretty_table(io, mx; tf = PrettyTables.tf_compact, header = names(mx))
end
