struct EstimateTable
    name::Vector{String}
    estimate::Vector{Float64}
    se::Vector{Float64}
    df::Vector{Float64}
    t::Vector{Float64}
    pval::Vector{Float64}
    level::Vector{Float64}
    cil::Vector{Float64}
    ciu::Vector{Float64}
end
# Стандартная ошибка контраста.
function contrast_se(lmm, l::AbstractVector)
    coefn(lmm) == lmm.rankx && return sqrt(dot(l, lmm.result.c, l))
    piv = view(lmm.pivotvec, 1:lmm.rankx)
    for i in 1:coefn(lmm)
        i in piv || iszero(l[i]) || return oftype(float(l[i]), NaN)
    end
    lr = view(l, piv)
    sqrt(dot(lr, lmm.result.c, lr))
end

function contrast_stats(est, se, df, level)
    (isfinite(se) && isfinite(df) && df > 0) && isfinite(est) || return (NaN, NaN, NaN, NaN)
    t    = abs(est / se)
    pval = ccdf(TDist(df), t) * 2
    d    = se * quantile(TDist(df), 1 - (1 - level) / 2)
    (t, pval, est - d, est + d)
end
"""
    estimate(lmm, l::AbstractVector; level = 0.95, name = "Estimate")

Estimate table for l vector. Satter DF used.

On a model of deficient rank the standard error is `NaN` only when `l` loads on
a dropped coefficient; otherwise it is the value [`stderror`](@ref) reports,
not the `NaN` that [`vcov`](@ref) would propagate. Where `se` or `df` is `NaN`,
`t`, `pval` and the interval are `NaN` too and no `TDist` is built.
"""
function estimate(lmm, l::AbstractVector; level = 0.95, name = "Estimate")
    est  = coef(lmm)'*l
    se   = contrast_se(lmm, l)
    df   = dof_satter(lmm, l)
    t, pval, cil, ciu = contrast_stats(est, se, df, level)
    EstimateTable([name], [est], [se], [df], [t], [pval], [level], [cil], [ciu])
end
"""
    estimate(lmm; level = 0.95)

Estimates table. Satter DF used.
"""
function estimate(lmm; level = 0.95)
    coe   = coef(lmm)
    l     = zeros(length(coe))
    vname = coefnames(lmm)
    vest  = Vector{Float64}(undef, length(coe))
    vse   = Vector{Float64}(undef, length(coe))
    vdf   = Vector{Float64}(undef, length(coe))
    vt    = Vector{Float64}(undef, length(coe))
    vpval = Vector{Float64}(undef, length(coe))
    vlevel= fill(level, length(coe))
    vcil  = Vector{Float64}(undef, length(coe))
    vciu  = Vector{Float64}(undef, length(coe))
    for i = 1:length(coe)
        fill!(l, 0)
        l[i]  = 1
        vest[i] = coe'*l
        vse[i]  = contrast_se(lmm, l)
        vdf[i]  = dof_satter(lmm, l)
        vt[i], vpval[i], vcil[i], vciu[i] = contrast_stats(vest[i], vse[i], vdf[i], level)
    end
    EstimateTable(vname, vest, vse, vdf, vt, vpval, vlevel, vcil, vciu)
end

function Base.show(io::IO, et::EstimateTable)
    println(io, "  Estiamte")
    mx = metida_table(et.name,  et.estimate, et.se, et.df, et.t, et.pval, et.level, et.cil, et.ciu; names = (:Name, :Estimate, :SE, :DF, :t, :pval, :Level, :LCI, :UCI))
    PrettyTables.pretty_table(io, mx; tf = PrettyTables.tf_compact)
end
