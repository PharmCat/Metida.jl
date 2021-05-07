#dof_contain.jl
function zmatrix(lmm::LMM, i)
    l   = zeros(Int, lmm.covstr.sn[i])'
    rzm = Matrix{eltype(lmm.covstr.z)}(undef, nobs(lmm), length(lmm.covstr.zrndur[i]) * lmm.covstr.sn[i])
    sn = 1
    for b = 1:length(lmm.covstr.vcovblock)
        zblock    = view(lmm.covstr.z, lmm.covstr.vcovblock[b], lmm.covstr.zrndur[i])
        for s = 1:length(lmm.covstr.sblock[b][i])
            zi    = view(zblock, lmm.covstr.sblock[b][i][s], :)
            l[sn] = 1
            copyto!(view(rzm, lmm.covstr.vcovblock[b][lmm.covstr.sblock[b][i][s]], :), kron(l, zi))
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
#=
function zmatrix(lmm, r)
        l   = zeros(Int, lmm.covstr.sn[r])
        si  = 1
        rzm = Matrix{Int}(undef, 0, length(lmm.covstr.zrndur[r]) * lmm.covstr.sn[r])
        for b = 1:length(lmm.covstr.vcovblock)
            zblock    = view(lmm.covstr.z, lmm.covstr.vcovblock[b], lmm.covstr.zrndur[r])
            for s = 1:length(lmm.covstr.sblock[b][r])
                zi    = view(zblock, lmm.covstr.sblock[b][r][s], :)
                l[si] = 1
                rzm   = vcat(rzm, kron(l', zi))
                l[si] = 0
                si   +=1
            end
        end
        rzm
end
=#
"""
    dof_contain(lmm)

!!! warning
    Experimental

Return the containment denominator degrees of freedom: rank(XZ) - rank(X)
"""
function dof_contain(lmm)
    rankxz(lmm) - lmm.rankx
end
#function dof_contain(lmm, i)
#    rank(hcat(lmm.data.xv, fullzmatrix(lmm))) - lmm.rankx
#end

#=
function dof_contain2(lmm)
    tl  = length(lmm.mf.f.rhs.terms)
    df  = Vector{Int}(undef, tl)
    dfb = Vector{Int}(undef, coefn(lmm))
    cnt = 1
    for i = 1:tl
        dfv=Vector{Int}(undef, 0)
        for r = 1:length(lmm.covstr.random)
            if contain(lmm.mf.f.rhs.terms[i], lmm.covstr.random[r])
                push!(dfv, rank(hcat(lmm.data.xv, zmatrix(lmm, r))))
            end
        end
        if length(dfv) > 0 df[i] = minimum(dfv) else df[i] =  nobs(lmm) - rank(hcat(lmm.data.xv, fullzmatrix(lmm))) end
        for c = 1:termsize(lmm.mf.f.rhs.terms[i])
            dfb[cnt] = df[i]
            cnt += 1
        end
    end
    dfb
end
=#
function dof_contain(lmm, i)

end
