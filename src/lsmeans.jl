#lsmeans.jl

################################################################################
#                           Least-squares means
################################################################################

"""
    LSMeansTable

The return type of [`lsmeans`](@ref).

| field | type | holds |
|---|---|---|
| `means` | [`EstimateTable`](@ref) | ``L\\hat\\beta``, SE, DF, t, p, interval |
| `pairs` | `EstimateTable` or `nothing` | row differences, with `pairwise = true` |
| `factors` | `Vector{Symbol}` | the factors of the margin |
| `levels` | `Vector{Vector{String}}` | their levels, per row |
| `n` | `Vector{Int}` | observations per row, `-1` when unknown |
| `estimable` | `Vector{Bool}` | ``L \\in \\mathrm{rowspace}(X)`` |
| `weights` | `Symbol` | the weighting used |
| `at` | `Dict{Symbol, Float64}` | covariate values |

`show` prints the `EstimateTable` columns, `N` unless every count is unknown,
and `Estimable` only when a row is not; what is omitted is stated in a line
below the table.

See also [`lsmeans`](@ref), [`lsml`](@ref), [`estimate`](@ref).
"""
struct LSMeansTable
    means::EstimateTable
    pairs::Union{Nothing, EstimateTable}
    factors::Vector{Symbol}
    levels::Vector{Vector{String}}
    n::Vector{Int}
    estimable::Vector{Bool}
    weights::Symbol
    at::Dict{Symbol, Float64}
end

# Components of a term, the zero-width intercept dropped.
#
# The filter names that one term instead of asking each component for its
# width. A component of zero width must not disappear from the decomposition:
# dropped, it leaves `prod(wid)` equal to the width of the term and the margin
# is then computed as though the component were 1, silently. Kept, it reaches
# `lsmvalue`, which refuses what it has no value for. `width` is also not
# defined for every term type across `StatsModels` versions.
lsmcomps(t) = [c for c in (isa(t, InteractionTerm) ? collect(t.terms) : Any[t])
               if !isa(c, InterceptTerm{false})]

# Every categorical term of the fixed part, by variable. A variable may appear
# in several terms; the contrast matrix is the same object in each, so the first
# one found answers for all.
function lsmfactors(lmm::LMM)
    d = Dict{Symbol, Any}()
    for t in lmm.f.rhs.terms, c in lsmcomps(t)
        isa(c, CategoricalTerm) && !haskey(d, c.sym) && (d[c.sym] = c)
    end
    d
end

# Value of a component that is neither a factor nor an intercept. A
# `ContinuousTerm` carries its own sample mean.
function lsmvalue(lmm::LMM, t, c, at::Dict{Symbol, Float64})
    if isa(c, ContinuousTerm)
        return get(at, c.sym, c.mean)
    end
    if t === c                                  # стоит отдельным термом
        asg = assign(lmm)
        i   = findfirst(==(c), lmm.f.rhs.terms)
        if !isnothing(i)
            col = findall(==(i), asg)
            length(col) == 1 && return mean(view(lmm.data.xv, :, col[1]))
        end
    end
    nothing
end

"""
    lsml(lmm::LMM, want::Dict{Symbol, String};
         at = Dict{Symbol, Float64}(), weights::Symbol = :equal)

Return the `L` of one marginal mean, a vector of length `coefn(lmm)`.

`want` gives the level of each requested factor, `at` the value of a covariate
that is not to be taken at its sample mean.

For a term with components ``c_1 … c_k`` of widths ``w_1 … w_k`` the block of
`L` is the product of the component weights, the first component varying
fastest:

```math
L[t]_{i_1 … i_k} = \\prod_{j=1}^{k} v_j[i_j]
```

where ``v_j`` is, for ``c_j``:

| component | ``v_j`` |
|---|---|
| intercept | ``1`` |
| categorical, in `want` at level ``r`` | ``M_j[r, :]`` |
| categorical, not in `want` | ``\\frac{1}{n_j}\\sum_r M_j[r, :]`` |
| continuous | ``\\bar x_j``, or `at[c_j]` |

``M_j`` is the contrast matrix of the term as `StatsModels` built it, so `L`
does not depend on the coding. `weights = :equal` is the third row above and
the only weighting implemented.

See also [`lsmeans`](@ref), [`estimate`](@ref).
"""
function lsml(lmm::LMM, want::Dict{Symbol, String};
              at::Dict{Symbol, Float64} = Dict{Symbol, Float64}(),
              weights::Symbol = :equal)
    weights === :equal || throw(ArgumentError("weights = :$weights is not implemented, " *
        "only :equal (equal weights over the levels of the other factors)"))
    L = zeros(coefn(lmm))
    k = 0
    for t in lmm.f.rhs.terms
        w = StatsModels.width(t)
        w == 0 && continue
        ws = Vector{Vector{Float64}}()
        for c in lsmcomps(t)
            if isa(c, CategoricalTerm)
                m = c.contrasts.matrix
                if haskey(want, c.sym)
                    lv = string.(c.contrasts.levels)
                    i  = findfirst(==(want[c.sym]), lv)
                    isnothing(i) && throw(ArgumentError("factor $(c.sym) has no level " *
                        "\"$(want[c.sym])\"; levels are $(lv)"))
                    push!(ws, collect(view(m, i, :)))
                else
                    push!(ws, [sum(view(m, :, j)) / size(m, 1) for j in axes(m, 2)])
                end
            elseif isa(c, InterceptTerm)
                push!(ws, [1.0])
            else
                v = lsmvalue(lmm, t, c, at)
                isnothing(v) && throw(ArgumentError("term $(t) has a component $(c) whose " *
                    "value at the margin is not available: a function term inside an " *
                    "interaction has no column of its own in the model matrix"))
                push!(ws, [v])
            end
        end
        wid = [length(x) for x in ws]
        prod(wid) == w || throw(ArgumentError("term $(t) does not decompose into " *
            "components: $(prod(wid)) weights for $(w) columns"))
        st = termstrides(wid)
        for ci in CartesianIndices(Tuple(wid))
            L[k + termcolumn(ci, st)] = prod(ws[j][ci[j]] for j in eachindex(ws))
        end
        k += w
    end
    L
end

# Столбцы X, выброшенные при неполном ранге: pivotvec перечисляет оставленные.
lsmdropped(lmm::LMM) = setdiff(1:coefn(lmm), view(lmm.pivotvec, 1:lmm.rankx))

"""
    lsmestimable(lmm::LMM, L, P)

`true` when ``L'\\beta`` is estimable: ``\\|L' - L'P\\| < 10^{-8}`` with
``P = X^{+}X``, and ``L_i = 0`` at every column `i` that the rank reduction
dropped.
"""
function lsmestimable(lmm::LMM, L, P)
    norm(L' .- L' * P) < 1e-8 || return false
    for i in lsmdropped(lmm)
        abs(L[i]) < 1e-10 || return false
    end
    true
end

# How many observations the combination of levels actually has. Read off X, not
# off the data, because the model does not keep the table: a row of X is at
# level ℓ of factor f when its entries in f's own columns equal row ℓ of f's
# contrast matrix. Needs f to stand in the model as a main effect - a factor
# that appears only inside an interaction has no columns that isolate it, and
# the count is then unknown, reported as -1.
function lsmcount(lmm::LMM, want::Dict{Symbol, String})
    trms = lmm.f.rhs.terms
    asg  = assign(lmm)
    X    = lmm.data.xv
    keep = trues(size(X, 1))
    for (f, lv) in want
        i = findfirst(t -> isa(t, CategoricalTerm) && t.sym == f, trms)
        isnothing(i) && return -1
        c    = trms[i]
        cols = findall(==(i), asg)
        length(cols) == size(c.contrasts.matrix, 2) || return -1
        j = findfirst(==(lv), string.(c.contrasts.levels))
        isnothing(j) && return -1
        row = view(c.contrasts.matrix, j, :)
        for r in axes(X, 1)
            keep[r] || continue
            # Сравнение с допуском, а не по равенству: элементы матрицы
            # контрастов у стандартных кодировок точны, но пользовательская
            # кодировка может их считать.
            keep[r] = all(k -> isapprox(X[r, cols[k]], row[k]; atol = 1e-10),
                          eachindex(cols))
        end
    end
    count(keep)
end

"""
    lsmeans(lmm::LMM, factors; pairwise::Bool = false, level::Real = 0.95,
            weights::Symbol = :equal, at = Dict{Symbol, Float64}())

Return an [`LSMeansTable`](@ref) of least-squares (estimated marginal) means,
one row per combination of the levels of `factors`.

```math
\\hat\\mu = L\\hat\\beta, \\quad
\\mathrm{SE} = \\sqrt{L'CL}, \\quad
C = (X'V^{-1}X)^{-1}
```

`L` comes from [`lsml`](@ref); the DF are Satterthwaite ([`dof_satter`](@ref)),
and `means` is the [`EstimateTable`](@ref) that [`estimate`](@ref) returns for
that `L`. With `pairwise = true`, `pairs` holds `L_i - L_j` for every pair of
rows, in the same table type.

`factors` is a `Symbol` or a vector of them. `level` is the interval level.
`at` sets covariates away from their sample mean, `at = Dict(:dose => 10.0)`.
`weights = :equal` is the only weighting implemented; `:proportional` throws.

Rows vary with the last named factor fastest.

`estimable` is [`lsmestimable`](@ref); a row that is not estimable carries
`NaN` through `means`.

`n` is the number of observations at that combination of levels, `0` for a cell
of the design that has none - which can still be estimable - and `-1` where no
column of `X` isolates a requested factor, as happens for a factor present only
inside an interaction.

A term that is a function of a column (`var^2`, `log(dose)`) is held at the mean
of its own column, ``\\overline{\\log(\\mathrm{dose})}`` rather than
``\\log(\\overline{\\mathrm{dose}})``. Inside an interaction such a term has no
column of its own and the call throws.

[`emmeans`](@ref) is an alias.

# Example

```julia
lmm = Metida.LMM(@formula(var ~ sequence + period + formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation|subject), Metida.DIAG))
Metida.fit!(lmm)
t = Metida.lsmeans(lmm, :formulation)
t.means.estimate
t.n
Metida.lsmeans(lmm, [:period, :formulation]; pairwise = true).pairs
```

# References

Searle, S. R., Speed, F. M., Milliken, G. A. (1980) Population marginal means
in the linear model: an alternative to least squares means. *The American
Statistician* 34, 216–221.

Goodnight, J. H., Harvey, W. R. (1978) Least squares means in the fixed effects
general linear model. SAS Technical Report R-103.
"""
function lsmeans(lmm::LMM, factors::AbstractVector{Symbol};
                 pairwise::Bool = false, level::Real = 0.95,
                 weights::Symbol = :equal,
                 at::Dict{Symbol, Float64} = Dict{Symbol, Float64}())
    isfitted(lmm) || error("Model not fitted!")
    isempty(factors) && throw(ArgumentError("no factors given"))
    length(unique(factors)) == length(factors) ||
        throw(ArgumentError("factors repeat: $(factors)"))

    fac = lsmfactors(lmm)
    for f in factors
        haskey(fac, f) || throw(ArgumentError("$(f) is not a factor of the fixed part; " *
            "its factors are $(sort(collect(keys(fac))))"))
    end
    for s in keys(at)
        any(c -> isa(c, ContinuousTerm) && c.sym == s,
            Iterators.flatten(lsmcomps(t) for t in lmm.f.rhs.terms)) ||
            throw(ArgumentError("$(s) given in `at` is not a covariate of the fixed part"))
    end

    levs = [string.(fac[f].contrasts.levels) for f in factors]
    P    = pinv(lmm.data.xv) * lmm.data.xv

    name = String[]; levels = Vector{String}[]
    est  = Float64[]; se = Float64[]; df = Float64[]; tv = Float64[]
    pv   = Float64[]; cil = Float64[]; ciu = Float64[]
    nn   = Int[];     ok = Bool[];     Ls = Vector{Float64}[]

    # Порядок строк: последний названный фактор меняется быстрее всех, как
    # его и читают - внутри period по formulation, а не наоборот.
    for rci in Iterators.product(reverse(levs)...)
        lv   = reverse(collect(String, rci))
        want = Dict(factors[i] => lv[i] for i in eachindex(factors))
        L    = lsml(lmm, want; at = at, weights = weights)
        push!(name, join([string(factors[i]) * ": " * lv[i] for i in eachindex(factors)], " & "))
        push!(levels, lv); push!(Ls, L)
        push!(nn, lsmcount(lmm, want))
        e = lsmestimable(lmm, L, P)
        push!(ok, e)
        if e
            t = estimate(lmm, L; level = level)
            push!(est, t.estimate[1]); push!(se, t.se[1]); push!(df, t.df[1])
            push!(tv, t.t[1]); push!(pv, t.pval[1])
            push!(cil, t.cil[1]); push!(ciu, t.ciu[1])
        else
            for v in (est, se, df, tv, pv, cil, ciu); push!(v, NaN); end
        end
    end

    prs = nothing
    if pairwise
        pn = String[]; pe = Float64[]; pse = Float64[]; pdf = Float64[]
        pt = Float64[]; pp = Float64[]; pl = Float64[]; pu = Float64[]
        for i in 1:length(Ls), j in (i + 1):length(Ls)
            nm = name[i] * " - " * name[j]
            push!(pn, nm)
            if ok[i] && ok[j] && lsmestimable(lmm, Ls[i] .- Ls[j], P)
                t = estimate(lmm, Ls[i] .- Ls[j]; level = level, name = nm)
                push!(pe, t.estimate[1]); push!(pse, t.se[1]); push!(pdf, t.df[1])
                push!(pt, t.t[1]); push!(pp, t.pval[1])
                push!(pl, t.cil[1]); push!(pu, t.ciu[1])
            else
                for v in (pe, pse, pdf, pt, pp, pl, pu); push!(v, NaN); end
            end
        end
        prs = EstimateTable(pn, pe, pse, pdf, pt, pp, fill(float(level), length(pn)), pl, pu)
    end

    mns = EstimateTable(name, est, se, df, tv, pv, fill(float(level), length(name)),
                        cil, ciu)
    LSMeansTable(mns, prs, collect(Symbol, factors), levels, nn, ok, weights, at)
end

lsmeans(lmm::LMM, factor::Symbol; kwargs...) = lsmeans(lmm, [factor]; kwargs...)

"""
    emmeans(lmm::LMM, factors; kwargs...)

Alias for [`lsmeans`](@ref); same arguments, same return type.
"""
emmeans(lmm::LMM, factors; kwargs...) = lsmeans(lmm, factors; kwargs...)

function Base.show(io::IO, t::LSMeansTable)
    m = t.means
    println(io, "  Least-squares means (weights = :", t.weights, ")")
    isempty(t.at) ||
        println(io, "  covariates held at: ",
                join([string(k) * " = " * string(v) for (k, v) in t.at], ", "))
    # Колонки EstimateTable, а дополнительные - только когда они что-то
    # говорят. Столбец из одних `true` или из одних «неизвестно» занимает
    # место и ничего не сообщает, поэтому не печатается, а сказанное про него
    # уходит в строку под таблицей.
    cols  = Any[m.name, m.estimate, m.se, m.df, m.t, m.pval, m.cil, m.ciu]
    names = Symbol[:Name, :Estimate, :SE, :DF, :t, :pval, :LCI, :UCI]
    known = !all(==(-1), t.n)
    if known
        push!(cols, t.n); push!(names, :N)
    end
    if any(!, t.estimable)
        push!(cols, t.estimable); push!(names, :Estimable)
    end
    mx = metida_table(cols...; names = Tuple(names))
    PrettyTables.pretty_table(io, mx; tf = PrettyTables.tf_compact)
    any(!, t.estimable) &&
        println(io, "  rows marked not estimable are outside the row space of X")
    known || println(io, "  the number of observations per row is unknown: a factor ",
                     "appears only inside an interaction")
    any(iszero, t.n) &&
        println(io, "  N = 0: no observations in that cell, the mean is an extrapolation")
    known && any(==(-1), t.n) &&
        println(io, "  N = -1: the count is unknown, which is not the same as zero")
    if !isnothing(t.pairs)
        println(io)
        println(io, "  Pairwise differences")
        show(io, t.pairs)
    end
    nothing
end
