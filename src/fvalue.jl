# fvalue.jl

#=
Metida.fvalue(lmm, [0 0 1 0 0 0; 0 0 0 1 0 0; 0 0 0 0 1 0])
=#
"""
    fvalue(lmm::LMM, l::Matrix)

F value for contrast matrix l.
"""
function fvalue(lmm::LMM, l::Matrix)
    lcl = l*lmm.result.c*l'
    f   = lmm.result.beta'*l'*pinv(lcl)*l*lmm.result.beta/rank(lcl)
end
