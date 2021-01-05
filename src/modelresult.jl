#modelresult.jl

mutable struct ModelResult
    fit::Bool
    optim
    theta
    reml
    beta
    h
    c
    se
    function ModelResult(
    optim,
    theta,
    reml,
    beta,
    h,
    c,
    se)
        new(true,
        optim,
        theta,
        reml,
        beta,
        h,
        c,
        se)
    end
    function ModelResult()
        new(false, nothing, nothing, nothing, nothing, nothing, nothing, nothing)
    end
end


function Base.show(io::IO, lmmr::ModelResult)
    println(io, "Theta :" , lmmr.theta)
    println(io, "Beta :" , lmmr.beta)
    println(io, "H :" , lmmr.h)
    println(io, "C :" , lmmr.c)
    println(io, "se :" , lmmr.se)
end
