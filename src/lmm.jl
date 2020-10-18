#lmm.jl

"""
    LMM{T} <: MetidaModel

    Linear-mixed model.
"""
struct LMM{T} <: MetidaModel
    model::FormulaTerm
    mf::ModelFrame
    mm::ModelMatrix
    covstr::CovStructure{T}
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
            xa, za, rza, ya = subjblocks(data, subject, mm.m, covstr.z, mf.data[mf.f.lhs.sym], covstr.repeated.model === nothing ? nothing : covstr.rz)
            lmmdata = LMMData(xa, za, rza, ya)
        else
            lmmdata = LMMData([mm.m], [covstr.z], [covstr.rz], [mf.data[mf.f.lhs.sym]])
        end
        new{eltype(mm.m)}(model, mf, mm, covstr, lmmdata, rank(mm.m), ModelResult())
    end
end
################################################################################


function Base.show(io::IO, lmm::LMM)
    println(io, "Linear Mixed Model: ", lmm.model)
    for i = 1:length(lmm.covstr.random)
        println(io, "Random $i: ")
        println(io, "   Model: $(lmm.covstr.random[i].model === nothing ? "nothing" : lmm.covstr.random[i].model)")
        println(io, "   Type: $(lmm.covstr.random[i].covtype.s) ($(lmm.covstr.t[i]))")
        println(io, "   Coefnames: $(coefnames(lmm.covstr.schema[i]))")
    end
    println(io, "Repeated: ")
    println(io, "   Model: $(lmm.covstr.repeated.model === nothing ? "nothing" : lmm.covstr.repeated.model)")
    println(io, "   Type: $(lmm.covstr.repeated.covtype.s) ($(lmm.covstr.t[end]))")
    println(io, "   Coefnames: $(lmm.covstr.repeated.model === nothing ? "-" : coefnames(lmm.covstr.schema[end]))")
    println(io, "")
    if lmm.result.fit
        print(io, "Status: ")
        Optim.converged(lmm.result.optim) ? printstyled(io, "converged \n"; color = :green) : printstyled(io, "not converged \n"; color = :red)
        println(io, "")
        println(io, "   -2 logREML: ", round(lmm.result.reml, sigdigits = 6))
        println(io, "")
        println(io, "   Fixed effects:")
        println(io, "")
        #chl = '─'
        mx  = hcat(coefnames(lmm.mf), round.(lmm.result.beta, sigdigits = 6), round.(lmm.result.se, sigdigits = 6))
        mx  = vcat(["Name" "Estimate" "SE"], mx)
        printmatrix(io, mx)
        println(io, "")
        println(io, "Random effects:")
        println(io, "")
        println(io, "   θ vector: ", round.(lmm.result.theta, sigdigits = 6))
        println(io, "")

        mx = hcat(Matrix{Any}(undef, lmm.covstr.tl, 1), lmm.covstr.rcnames, lmm.covstr.ct, round.(lmm.result.theta, sigdigits = 6))

        for i = 1:length(lmm.covstr.random)
            view(mx, lmm.covstr.tr[i], 1) .= "Random $i"
        end
        view(mx, lmm.covstr.tr[end], 1) .= "Repeated"
        for i = 1:lmm.covstr.tl
            if mx[i, 3] == :var mx[i, 4] = round.(mx[i, 4]^2, sigdigits = 6) end
        end
        printmatrix(io, mx)
    else
        println(io, "Not fitted.")
    end
end
