#modelresult.jl

mutable struct ModelResult
    fit::Bool
    optim
    theta::Vector{Float64}
    reml::Float64
    beta::Vector{Float64}
    h::Union{Matrix, Nothing}
    c::Matrix{Float64}
    se::Vector{Float64}
    grc::Union{Vector, Nothing}
    ipd::Bool
    function ModelResult(fit, optim, theta, reml, beta, h, c, se, grc, idp)
        new(fit, optim, theta, reml, beta, h, c, se, grc, idp)
    end
    function ModelResult()
        ModelResult(false, nothing, nothing, NaN, nothing, nothing, nothing, nothing, nothing, false)
    end
end

function Base.show(io::IO, lmmr::ModelResult)
    println(io, "Theta :" , lmmr.theta)
    println(io, "Beta :" , lmmr.beta)
    println(io, "H :" , lmmr.h)
    println(io, "C :" , lmmr.c)
    println(io, "se :" , lmmr.se)
end
