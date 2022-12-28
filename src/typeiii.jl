
struct ContrastTable
    name::Vector{String}
    f::Vector{Float64}
    ndf::Vector{Float64}
    df::Vector{Float64}
    pval::Vector{Float64}
end
"""
    typeiii(lmm::LMM{T}; ddf::Symbol = :satter) where T

!!! warning
    Experimental

Type III table.
"""
function typeiii(lmm::LMM; ddf::Symbol = :satter)
    if !isfitted(lmm) error("Model not fitted!") end
    c           = length(lmm.mf.f.rhs.terms)
    d           = Vector{Int}(undef, 0)
    fac         = Vector{String}(undef, c)
    F           = Vector{Float64}(undef,c)
    df          = Vector{Float64}(undef, c)
    ndf         = Vector{Float64}(undef, c)
    pval        = Vector{Float64}(undef, c)
    for i = 1:c
        iterm = lmm.mf.f.rhs.terms[i]
        
        if typeof(iterm) <: InterceptTerm{false}
            push!(d, i)
            fac[i] = ""
            continue
        end
        
        fac[i]  = tname(iterm) 
        L       = lcontrast(lmm, i)
        F[i]    = fvalue(lmm, L)
        ndf[i]  = rank(L)
        if ddf == :satter
            df[i]  = dof_satter(lmm, L)
        elseif ddf == :contain
            df[i]  = dof_contain_f(lmm, i)
        elseif ddf == :residual
            df[i]  = dof_residual(lmm)
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

User contrast table.
ddf = `:satter` or `:residual` or any number for direct ddf setting.
"""
function contrast(lmm, l::AbstractMatrix; name::String = "Contrast", ddf = :satter)
    if !isfitted(lmm) error("Model not fitted!") end
    if lmm.rankx != size(l, 2) error("size(l, 2) not equal rank X!") end
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
    ContrastTable([name], [F], [ndf], [df], [pval])
end

function Base.show(io::IO, at::ContrastTable)
    mx = metida_table(at.name,  at.f, at.ndf, at.df, at.pval; names = (:Name, :F, :ndf, :ddf, :pval))
    PrettyTables.pretty_table(io, mx; tf = PrettyTables.tf_compact, header = names(mx))
end
