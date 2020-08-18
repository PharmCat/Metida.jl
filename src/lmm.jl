#lmm.jl

struct LMM{T} <: MetidaModel
    model::FormulaTerm
    mf::ModelFrame
    mm::ModelMatrix
    function LMM(model, data; contrasts=Dict{Symbol,Any}())
        mf = ModelFrame(model, data; contrasts = contrasts)
        mm = ModelMatrix(mf)
        new{eltype(mm.m)}(model, mf, mm)
    end
end


function Base.show(io::IO, lmm::LMM)
    println(io, "Linear Mixed Model: ", lmm.model)
end
