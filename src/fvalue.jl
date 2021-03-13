# fvalue.jl


function fvalue(lmm::LMM, l::Matrix)
    lcl = l*lmm.result.c*l'
    f   = lmm.result.beta'*l'*pinv(lcl)*l*lmm.result.beta/rank(lcl)
end
