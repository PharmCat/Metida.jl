#lmm.jl

struct LMMLogMsg
    type::Symbol
    msg::String
end
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
    nfixed::Int
    rankx::UInt32
    result::ModelResult
    blocksolve::Bool
    log::Vector{LMMLogMsg}

    function LMM(model, data; contrasts=Dict{Symbol,Any}(), subject::Union{Nothing, Symbol, AbstractVector{Symbol}} = nothing,  random::Union{Nothing, VarEffect, Vector{VarEffect}} = nothing, repeated::Union{Nothing, VarEffect} = nothing)
        #need check responce - Float
        if repeated === nothing && random === nothing
            error("No effects specified!")
        end
        if isa(subject, Nothing)
            subject = Vector{Symbol}(undef, 0)
        elseif isa(subject, Symbol)
            subject = [subject]
        end
        lmmlog = Vector{LMMLogMsg}(undef, 0)
        mf     = ModelFrame(model, data; contrasts = contrasts)
        mm     = ModelMatrix(mf)
        nfixed = nterms(mf)
        if repeated === nothing
            repeated = VarEffect(Metida.@covstr(1), Metida.ScaledIdentity(), subj = intersectsubj(random))
        end
        if random === nothing
            random = VarEffect(Metida.@covstr(0), Metida.RZero(), subj = repeated.subj)
        end
        if !isa(random, Vector) random = [random] end
        #blocks
        intsub, eq = intersectsubj(random, repeated)
        blocksolve = false
        if length(subject) > 0 blocksolve = true end
        if eq blocksolve = true end
        if (length(subject) > 0 && !eq && length(intsub) > 0) || (length(subject) > 0 && !issetequal(subject, intsub) && length(intsub) > 0)
            lmmlog!(lmmlog, 1, LMMLogMsg(:WARN, "Global subject variable is specified, but variance effect(s) have different subject's values!"))
        end
        if length(subject) == 0
            subject = intsub
        end
        block  = intersectdf(data, subject)
        lmmdata = LMMData(mm.m, response(mf), block, subject)
        covstr = CovStructure(random, repeated, data, block)
        new{eltype(mm.m)}(model, mf, mm, covstr, lmmdata, nfixed, rank(mm.m), ModelResult(), blocksolve, lmmlog)
    end
end
################################################################################
"""
    thetalength(lmm::LMM)

Length of theta vector.
"""
function thetalength(lmm)
    lmm.covstr.tl
end

"""
    theta(lmm::LMM)

Return theta vector.
"""
function theta(lmm::LMM)
    copy(lmm.result.theta)
end
"""
    rankx(lmm::LMM)

Return rank of X matrix.
"""
function rankx(lmm::LMM)
    Int(lmm.rankx)
end
################################################################################
function lmmlog!(io, lmmlog::Vector{LMMLogMsg}, verbose, vmsg)
    if verbose == 1
        push!(lmmlog, vmsg)
    elseif verbose == 2
        println(io, vmsg)
        push!(lmmlog, vmsg)
    elseif verbose == 3
        if vmsg.type == :ERROR println(io, vmsg) end
        push!(lmmlog, vmsg)
    end
end
function lmmlog!(lmmlog::Vector{LMMLogMsg}, verbose, vmsg)
    lmmlog!(stdout, lmmlog, verbose, vmsg)
end
function lmmlog!(io, lmm::LMM, verbose, vmsg)
    lmmlog!(io, lmm.log, verbose, vmsg)
end
function lmmlog!(lmm::LMM, verbose, vmsg)
    lmmlog!(stdout, lmm, verbose, vmsg)
end
function lmmlog!(lmm::LMM, vmsg)
    lmmlog!(stdout, lmm, 1, vmsg)
end
################################################################################

function Base.show(io::IO, lmm::LMM)
    println(io, "Linear Mixed Model: ", lmm.model)
    for i = 1:length(lmm.covstr.random)
        println(io, "Random $i: ")
        if lmm.covstr.random[i].covtype.s == :ZERO
            println(io, "   No")
            continue
        end
        println(io, "   Model: $(lmm.covstr.random[i].model === nothing ? "nothing" : lmm.covstr.random[i].model)")
        println(io, "   Type: $(lmm.covstr.random[i].covtype.s) ($(lmm.covstr.t[i]))")
        #println(io, "   Coefnames: $(coefnames(lmm.covstr.schema[i]))")
    end
    println(io, "Repeated: ")
    println(io, "   Model: $(lmm.covstr.repeated.model === nothing ? "nothing" : lmm.covstr.repeated.model)")
    println(io, "   Type: $(lmm.covstr.repeated.covtype.s) ($(lmm.covstr.t[end]))")
    #println(io, "   Coefnames: $(lmm.covstr.repeated.model === nothing ? "-" : coefnames(lmm.covstr.schema[end]))")
    println(io, "")
    if lmm.result.fit
        print(io, "Status: ")
        printresult(io, lmm.result.optim)
        #Optim.converged(lmm.result.optim) ? printstyled(io, "converged \n"; color = :green) : printstyled(io, "not converged \n"; color = :red)
        #if length(lmm.log) > 0  printstyled(io, "Warnings! See lmm.log \n"; color = :yellow) end
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
            if lmm.covstr.random[i].covtype.s != :ZERO
                view(mx, lmm.covstr.tr[i], 1) .= "Random $i"
            end
        end
        view(mx, lmm.covstr.tr[end], 1) .= "Residual"
        for i = 1:lmm.covstr.tl
            if mx[i, 3] == :var mx[i, 4] = round.(mx[i, 4]^2, sigdigits = 6) end
        end
        printmatrix(io, mx)
    else
        println(io, "Not fitted.")
    end
end

function printresult(io, res::T) where T <: Optim.MultivariateOptimizationResults
    Optim.converged(res) ? printstyled(io, "converged \n"; color = :green) : printstyled(io, "not converged \n"; color = :red)
end
function printresult(io, res)
    if res[3] == :FTOL_REACHED || res[3] == :XTOL_REACHED || res[3] == :SUCCESS
        printstyled(io, "converged ($(string(res[3])))\n"; color = :green)
    else
        printstyled(io, "not converged ($(string(res[3])))\n"; color = :red)
    end
end

function Base.show(io::IO, lmmlog::LMMLogMsg)
    if lmmlog.type == :INFO
        printstyled(io, "  INFO  : "; color = :blue)
        println(io, lmmlog.msg)
    elseif lmmlog.type == :WARN
        printstyled(io, "  WARN  : "; color = :yellow)
        println(io, lmmlog.msg)
    elseif lmmlog.type == :ERROR
        printstyled(io, "  ERROR : "; color = :red)
        println(io, lmmlog.msg)
    end
end
