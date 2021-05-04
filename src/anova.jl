
struct ANOVATable
    name::Vector
    f::Vector
    ndf::Vector
    df::Vector
    pval::Vector
end
"""
    anova(lmm::LMM)

!!! warning
    Experimental

Type III table.
"""
function anova(lmm::LMM{T}; ddf::Symbol = :satter) where T
    if !isfitted(lmm) error("Model not fitted!") end
    c           = nterms(lmm.mf)
    d           = Vector{Int}(undef, 0)
    fac         = Vector{String}(undef, c)
    F           = Vector{T}(undef,c)
    df          = Vector{T}(undef, c)
    ndf         = Vector{T}(undef, c)
    pval        = Vector{T}(undef, c)
    for i = 1:c
        if typeof(lmm.mf.f.rhs.terms[i]) <: InterceptTerm{true}
            fac[i] = "(Intercept)"
        elseif typeof(lmm.mf.f.rhs.terms[i]) <: InterceptTerm{false}
            push!(d, i)
            continue
        else
            fac[i] = string(lmm.mf.f.rhs.terms[i].sym)
        end
        L       = lcontrast(lmm, i)
        F[i]    = fvalue(lmm, L)
        ndf[i]  = rank(L)
        if ddf == :satter
            df[i]  = dof_satter(lmm, L)
        elseif ddf == :contain
            df[i]  = dof_contain(lmm, i)
        elseif ddf == :residual
            df[i]  = dof_residual(lmm, i)
        end
        pval[i] = ccdf(FDist(ndf[i], df[i]), F[i])
    end
    if length(d) > 0
        deleteat!(fac, d)
        deleteat!(F, d)
        deleteat!(ndf, d)
        deleteat!(df, d)
        deleteat!(pval, d)
    end
    ANOVATable(fac, F, ndf, df, pval)
end

function Base.show(io::IO, at::ANOVATable)
    println(io, "  Type III Tests of Fixed Effects")
    mx = hcat(at.name,  round.(at.f; digits = 4), round.(at.ndf; digits = 4), round.(at.df; digits = 4), round.(at.pval; digits = 4))
    mx = vcat(["Name" "F" "ndf" "ddf" "pval"], mx)
    printmatrix(io, mx; header = true)
end
