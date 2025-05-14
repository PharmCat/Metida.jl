# fvalue.jl

"""
    fvalue(lmm::LMM, l::AbstractMatrix)

F value for contrast matrix `l`.

```math
F = \\frac{\\beta'L'(LCL')^{-1}L\\beta}{rank(LCL')}
```
"""
function fvalue(lmm::LMM, l::AbstractMatrix)
    fvalue_(lmm, ifelse(lmm.rankx == coefn(lmm), l, view(l, :, lmm.pivotvec)))
end
function fvalue_(lmm::LMM, l::AbstractMatrix)
    lcl = l*lmm.result.c*l'
    f   = lmm.result.beta'*l'*pinv(lcl)*l*lmm.result.beta/rank(lcl)
end
