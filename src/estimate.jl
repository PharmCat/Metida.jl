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
"""
    estimate(lmm, l::AbstractVector; level = 0.95, name = "Estimate")

Estimate table for l vector. Satter DF used.
"""
function estimate(lmm, l::AbstractVector; level = 0.95, name = "Estimate")
    est  = coef(lmm)'*l
    #se   = sqrt(mulαtβα(l, vcov(lmm)))
    se   = sqrt(dot(l, vcov(lmm), l))
    df   = dof_satter(lmm, l)
    t    = abs(est/se)
    pval = ccdf(TDist(df), t)*2
    d = se*quantile(TDist(df), 1-(1-level)/2)
    cil = est - d
    ciu = est + d
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
        #vse[i]  = sqrt(mulαtβα(l, vcov(lmm)))
        vse[i]  = sqrt(dot(l, vcov(lmm), l))
        vdf[i]  = dof_satter(lmm, l)
        vt[i]   = abs(vest[i]/vse[i])
        vpval[i]= ccdf(TDist(vdf[i]), vt[i])*2
        d       = vse[i]*quantile(TDist(vdf[i]), 1-(1-level)/2)
        vcil[i] = vest[i] - d
        vciu[i] = vest[i] + d
    end
    EstimateTable(vname, vest, vse, vdf, vt, vpval, vlevel, vcil, vciu)
end

function Base.show(io::IO, et::EstimateTable)
    println(io, "  Estiamte")
    mx = metida_table(et.name,  et.estimate, et.se, et.df, et.t, et.pval, et.level, et.cil, et.ciu; names = (:Name, :Estimate, :SE, :DF, :t, :pval, :Level, :LCI, :UCI))
    PrettyTables.pretty_table(io, mx; tf = PrettyTables.tf_compact)
end
