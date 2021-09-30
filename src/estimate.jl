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

function estimate(lmm, l::AbstractVector; level = 0.95, name = "Estimate")
    est  = coef(lmm)'*l
    se   = sqrt(mulαtβα(l, vcov(lmm)))
    df   = dof_satter(lmm, l)
    t    = abs(est/se)
    pval = ccdf(TDist(df), t)*2
    d = se*quantile(TDist(df), 1-(1-level)/2)
    cil = est - d
    ciu = est + d
    EstimateTable([name], [est], [se], [df], [t], [pval], [level], [cil], [ciu])
end

function Base.show(io::IO, et::EstimateTable)
    println(io, "  Estiamte")
    mx = metida_table(et.name,  et.estimate, et.se, et.df, et.t, et.pval, et.level, et.cil, et.ciu; names = (:Name, :Estimate, :SE, :DF, :t, :pval, :Level, :LCI, :UCI))
    PrettyTables.pretty_table(io, mx; tf = PrettyTables.tf_compact)
end
