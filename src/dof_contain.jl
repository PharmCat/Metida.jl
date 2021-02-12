#dof_contain.jl

function fullzmatrix(lmm)
    fzm = Matrix{Int}(undef, nobs(lmm), 0)
    for r = 1:length(lmm.covstr.random)
        l   = zeros(Int, lmm.covstr.sn[r])
        si  = 1
        rzm = Matrix{Int}(undef, 0, length(lmm.covstr.zrndur[r]) * lmm.covstr.sn[r])
        for b = 1:length(lmm.data.block)
            zblock    = view(lmm.covstr.z, lmm.data.block[b], lmm.covstr.zrndur[r])
            for s = 1:length(lmm.covstr.sblock[b][r])
                zi    = view(zblock, lmm.covstr.sblock[b][r][s], :)
                l[si] = 1
                rzm   = vcat(rzm, kron(l', zi))
                l[si] = 0
                si   +=1
            end
        end
        fzm = hcat(fzm, rzm)
    end
    fzm
end

function dof_contain(lmm)
    rank(hcat(lmm.data.xv, fullzmatrix(lmm))) - lmm.rankx
end
