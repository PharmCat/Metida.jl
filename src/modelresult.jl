#modelresult.jl

mutable struct ModelResult
    optim
    theta
    reml
    beta
    h
    c
    se
    function ModelResult(optim,
    theta,
    reml,
    beta,
    h,
    c,
    se)
        new(optim,
        theta,
        reml,
        beta,
        h,
        c,
        se)
    end
    function ModelResult()
        new(nothing, nothing, nothing, nothing, nothing, nothing, nothing)
    end
end
