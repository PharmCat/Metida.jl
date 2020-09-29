#lmm.jl

#Metida.LMM(@formula(var ~ formulation + period), df6)

struct LMM{T} <: MetidaModel
    model::FormulaTerm
    mf::ModelFrame
    mm::ModelMatrix
    covstr::CovStructure
    data::LMMData{T}
    rankx::Int
    result::ModelResult

    function LMM(model, data; contrasts=Dict{Symbol,Any}(), subject = nothing,  random = nothing, repeated = nothing)
        mf = ModelFrame(model, data; contrasts = contrasts)
        mm = ModelMatrix(mf)
        if random === nothing
            random = VarEffect()
        end
        if repeated === nothing
            repeated = VarEffect()
        end
        if !isa(random, Vector) random = [random] end
        covstr = CovStructure(random, repeated, data)
        if isa(subject, Symbol)
            xa, za, rza, ya = subjblocks(data, subject, mm.m, covstr.z, mf.data[mf.f.lhs.sym], covstr.rz)
            lmmdata = LMMData(xa, za, rza, ya)
        else
            lmmdata = LMMData([mm.m], [covstr.z], [Matrix{eltype(mm.m)}(undef,0,0)], [mf.data[mf.f.lhs.sym]])
        end
        new{eltype(mm.m)}(model, mf, mm, covstr, lmmdata, rank(mm.m), ModelResult())
    end
end


function Base.show(io::IO, lmm::LMM)
    println(io, "Linear Mixed Model: ", lmm.model)
end
