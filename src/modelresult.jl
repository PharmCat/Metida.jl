#modelresult.jl

mutable struct ModelResult
    fit::Bool
    optim
    theta::Union{Vector, Nothing}
    reml::Float64
    beta::Union{Vector, Nothing}
    h::Union{Matrix, Nothing}
    c::Union{Matrix, Nothing}
    se::Union{Vector, Nothing}
    grc::Union{Vector, Nothing}
    ipd::Bool
    function ModelResult()
        new(false, nothing, nothing, NaN, nothing, nothing, nothing, nothing, nothing, false)
    end
    #function ModelResult(tn::Int, bn::Int)
    #    new(false, nothing, Vector{Float64}(undef, tn), NaN, Vector{Float64}(undef, bn), Matrix{Float64}(undef, tn, tn),  Matrix{Float64}(undef, bn, bn), Vector{Float64}(undef, bn), nothing, false)
    #end
end


function Base.show(io::IO, lmmr::ModelResult)
    println(io, "Theta :" , lmmr.theta)
    println(io, "Beta :" , lmmr.beta)
    println(io, "H :" , lmmr.h)
    println(io, "C :" , lmmr.c)
    println(io, "se :" , lmmr.se)
end
