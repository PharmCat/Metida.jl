# fvalue.jl

"""
    fvalue(lmm::LMM, l::Matrix)

F value for contrast matrix `l`.

```math
F = \\frac{\\beta'L'(LCL')^{-1}L\\beta}{rank(LCL')}
```
"""
function fvalue(lmm::LMM, l::Matrix)
    lcl = l*lmm.result.c*l'
    f   = lmm.result.beta'*l'*pinv(lcl)*l*lmm.result.beta/rank(lcl)
end
