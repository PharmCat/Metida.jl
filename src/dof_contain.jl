#dof_contain.jl
function zmatrix(lmm::LMM, i)
    l   = zeros(Int, lmm.covstr.sn[i])'
    rzm = Matrix{eltype(lmm.covstr.z)}(undef, nobs(lmm), length(lmm.covstr.zrndur[i]) * lmm.covstr.sn[i])
    sn = 1
    for b = 1:length(lmm.covstr.vcovblock)
        zblock    = view(lmm.covstr.z, lmm.covstr.vcovblock[b], lmm.covstr.zrndur[i])
        for s = 1:subjn(lmm.covstr, i, b)
            suji  = getsubj(lmm.covstr, i, b, s)
            zi    = view(zblock, suji, :)
            l[sn] = 1
            copyto!(view(rzm, lmm.covstr.vcovblock[b][suji], :), kron(l, zi))
            l[sn] = 0
            sn   += 1
        end
    end
    rzm
end
function fullzmatrix(lmm)
    fzm = Matrix{Int}(undef, nobs(lmm), 0)
    for r = 1:length(lmm.covstr.random)
        rzm = zmatrix(lmm, r)
        fzm = hcat(fzm, rzm)
    end
    fzm
end

function rankxz(lmm::LMM)
    rank(hcat(lmm.data.xv, fullzmatrix(lmm)))
end
function rankxz(lmm::LMM, i)
    rank(hcat(lmm.data.xv, zmatrix(lmm, i)))
end

"""
    dof_contain(lmm, i)

!!! warning
    Experimental!
    Compute rank(XZi) for each random effect that syntactically contain factor assigned for β[i] element (Where Zi - Z matrix for random effect i).
    Minimum returned. If no random effect found  N - rank(XZ) returned.
"""
function dof_contain(lmm, i)
    ind  = lmm.modstr.assign[i]
    sym  = StatsModels.termvars(lmm.f.rhs.terms[ind])
    rr   = Vector{Int}(undef, 0)
    for r = 1:length(lmm.covstr.random)
        if length(intersect(sym, StatsModels.termvars(lmm.covstr.random[r].model))) > 0
            push!(rr, rankxz(lmm, r))
        end
    end
    if length(rr) > 0
        return minimum(rr)
    else
        return nobs(lmm) - rankxz(lmm)
    end
end

function dof_contain(lmm)
    dof   = zeros(Int, length(lmm.modstr.assign))
    rrt   = zeros(Int, length(lmm.covstr.random))
    rz   = 0
    for i = 1:length(lmm.modstr.assign)
        ind  = lmm.modstr.assign[i]
        sym  = StatsModels.termvars(lmm.f.rhs.terms[ind])
        rr   = Vector{Int}(undef, 0)
        for r = 1:length(lmm.covstr.random)
            if length(intersect(sym, StatsModels.termvars(lmm.covstr.random[r].model))) > 0
                if rrt[r] == 0 rrt[r] = rankxz(lmm, r) end
                push!(rr, rrt[r])
            end
        end
        if length(rr) > 0
            dof[i] = minimum(rr)
        else
            if rz == 0 rz = nobs(lmm) - rankxz(lmm) end
            dof[i] = rz
        end
    end
    dof
end
"""
    dof_contain_f(lmm, i)

!!! warning
    Experimental

"""
function dof_contain_f(lmm, i)
    sym  = StatsModels.termvars(lmm.f.rhs.terms[i])
    rr   = Vector{Int}(undef, 0)
    for r = 1:length(lmm.covstr.random)
        if length(intersect(sym, StatsModels.termvars(lmm.covstr.random[r].model))) > 0
            push!(rr, rankxz(lmm, r))
        end
    end
    if length(rr) > 0
        return minimum(rr)
    else
        return nobs(lmm) - rankxz(lmm)
    end
end
