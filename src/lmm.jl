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
    for i = 1:length(lmm.covstr.random)
        println(io, "Random $i: ")
        println(io, "   Model: $(lmm.covstr.random[i].model === nothing ? "nothing" : lmm.covstr.random[i].model)")
        println(io, "   Type: $(nameof(typeof(lmm.covstr.random[i].covtype))) ($(lmm.covstr.t[i]))")
        println(io, "   Coefnames: $(coefnames(lmm.covstr.schema[i]))")
    end
    println(io, "Repeated: ")
    println(io, "   Model: $(lmm.covstr.repeated.model === nothing ? "nothing" : lmm.covstr.repeated.model)")
    println(io, "   Type: $(nameof(typeof(lmm.covstr.repeated.covtype))) ($(lmm.covstr.t[end]))")
    println(io, "   Coefnames: $(coefnames(lmm.covstr.schema[end]))")
    if lmm.result.fit
        print("Status: ")
        Optim.converged(lmm.result.optim) ? printstyled(io, "converged \n"; color = :green) : printstyled(io, "not converged \n"; color = :red)
    else
        println(io, "Not fitted.")
    end
end
