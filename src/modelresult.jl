#modelresult.jl

mutable struct ModelResult
    fit::Bool
    optim
    theta::Union{Vector, Nothing}
    reml::Union{Float64, Nothing}
    beta::Union{Vector, Nothing}
    h::Union{Matrix, Nothing}
    c::Union{Matrix, Nothing}
    se::Union{Vector, Nothing}
    grc::Union{Vector, Nothing}

    function ModelResult()
        new(false, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing)
    end
end


function Base.show(io::IO, lmmr::ModelResult)
    println(io, "Theta :" , lmmr.theta)
    println(io, "Beta :" , lmmr.beta)
    println(io, "H :" , lmmr.h)
    println(io, "C :" , lmmr.c)
    println(io, "se :" , lmmr.se)
end
