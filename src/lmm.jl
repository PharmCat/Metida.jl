#lmm.jl

"""
    LMM(model, data; contrasts=Dict{Symbol,Any}(), subject::Union{Nothing, Symbol} = nothing,  random::Union{Nothing, VarEffect, Vector{VarEffect}} = nothing, repeated::Union{Nothing, VarEffect} = nothing)

Make Linear-Mixed Model object.

`model`: is a fixed-effect model (`@formula`)

`data`: tabular data

`contrasts`: contrasts for fixed factors

`random`: vector of random effects or single random effect

`repeated`: is a repeated effect (only one)

`subject`: is a block-diagonal factor
"""
struct LMM{T} <: MetidaModel
    model::FormulaTerm
    mf::ModelFrame
    mm::ModelMatrix
    covstr::CovStructure{T}
    data::LMMData{T}
    rankx::UInt32
    result::ModelResult
    blocksolve::Bool
    warn::Vector{String}

    function LMM(model, data; contrasts=Dict{Symbol,Any}(), subject::Union{Nothing, Symbol, AbstractVector{Symbol}} = nothing,  random::Union{Nothing, VarEffect, Vector{VarEffect}} = nothing, repeated::Union{Nothing, VarEffect} = nothing)
        if isa(subject, Nothing)
            subject = Vector{Symbol}(undef, 0)
        elseif isa(subject, Symbol)
            subject = [subject]
        elseif isa(subject,  AbstractVector{Symbol})
            #
        else
            throw(ArgumentError("subject type should be Symbol or Vector{tymbol}"))
        end
        warn = Vector{String}(undef, 0)
        mf   = ModelFrame(model, data; contrasts = contrasts)
        mm   = ModelMatrix(mf)
        if random === nothing
            random = VarEffect()
        end
        if repeated === nothing
            repeated = VarEffect()
        end
        if !isa(random, Vector) random = [random] end
        #blocks
        intsub, eq = intersectsubj(random, repeated)
        blocksolve = false
        if length(subject) > 0 blocksolve = true end
        if eq blocksolve = true end
        if (length(subject) > 0 && !eq && length(intsub) > 0) || (length(subject) > 0 && !issetequal(subject, intsub) && length(intsub) > 0) push!(warn, "::You specify global subject variable, but variance effect have different subject's values and would be ignored!") end
        if length(subject) == 0
            subject = intsub
        end
        block  = intersectdf(data, subject)
        lmmdata = LMMData(mm.m, mf.data[mf.f.lhs.sym], block, subject)
        covstr = CovStructure(random, repeated, data, block)
        new{eltype(mm.m)}(model, mf, mm, covstr, lmmdata, rank(mm.m), ModelResult(), blocksolve, warn)
    end
end
################################################################################
"""
    thetalength(lmm::LMM)

Length of theta vector.
"""
function thetalength(lmm::LMM)
    lmm.covstr.tl
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
        if length(lmm.warn) > 0  printstyled(io, "Warnings! See lmm.warn \n"; color = :yellow) end
        println(io, "")
        println(io, "   -2 logREML: ", round(lmm.result.reml, sigdigits = 6))
        println(io, "")
        println(io, "   Fixed effects:")
        println(io, "")
        #chl = '─'
        z = lmm.result.beta ./ lmm.result.se
        mx  = hcat(coefnames(lmm.mf), round.(lmm.result.beta, sigdigits = 6), round.(lmm.result.se, sigdigits = 6), round.(z, sigdigits = 6), round.(ccdf.(Chisq(1), abs2.(z)), sigdigits = 6))
        mx  = vcat(["Name" "Estimate" "SE" "z" "Pr(>|z|)"], mx)
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
