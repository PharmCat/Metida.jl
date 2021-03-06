#lmm.jl

struct LMMLogMsg
    type::Symbol
    msg::String
end
"""
    LMM(model, data; contrasts=Dict{Symbol,Any}(),  random::Union{Nothing, VarEffect, Vector{VarEffect}} = nothing, repeated::Union{Nothing, VarEffect} = nothing)

Make Linear-Mixed Model object.

`model`: is a fixed-effect model (`@formula`)

`data`: tabular data

`contrasts`: contrasts for fixed factors

`random`: vector of random effects or single random effect

`repeated`: is a repeated effect (only one)
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
    log::Vector{LMMLogMsg}

    function LMM(model, data; contrasts=Dict{Symbol,Any}(),  random::Union{Nothing, VarEffect, Vector{VarEffect}} = nothing, repeated::Union{Nothing, VarEffect} = nothing)
        #need check responce - Float
        if repeated === nothing && random === nothing
            error("No effects specified!")
        end

        lmmlog = Vector{LMMLogMsg}(undef, 0)
        mf     = ModelFrame(model, data; contrasts = contrasts)
        mm     = ModelMatrix(mf)
        nfixed = nterms(mf)
        if repeated === nothing
            repeated = VarEffect(Metida.@covstr(1|1), Metida.ScaledIdentity())
        end
        if random === nothing
            random = VarEffect(Metida.@covstr(0|1), Metida.RZero())
        end
        if !isa(random, Vector) random = [random] end

        if repeated.covtype.s == :SI && !isa(repeated.model, ConstantTerm)
            lmmlog!(lmmlog, 1, LMMLogMsg(:WARN, "Repeated effect not a constant, but covariance type is SI. "))
        end
        lmmdata = LMMData(mm.m, response(mf))
        covstr = CovStructure(random, repeated, data)
        new{eltype(mm.m)}(model, mf, mm, covstr, lmmdata, nfixed, rank(mm.m), ModelResult(), lmmlog)
    end
end
"""
    lcontrast(lmm::LMM, i::Int)

L-contrast matrix for `i` fixed effect.
"""
function lcontrast(lmm::LMM, i::Int)
    n = nterms(lmm.mf)
    if i > n || n < 1 error("Factor number out of range 1-$(n)") end
    inds = findall(x -> x==i, lmm.mm.assign)
    mx = zeros(length(inds), size(lmm.mm.m, 2))
    for i = 1:length(inds)
        mx[i, inds[i]] = 1
    end
    mx
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
    coefn(lmm)

Coef number.
"""
function coefn(lmm)
    length(lmm.result.beta)
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

Return rank of `X` matrix.
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
#MetidaNLopt use this
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
        println(io, "    Model: $(lmm.covstr.random[i].model === nothing ? "nothing" : string(lmm.covstr.random[i].model, "|", lmm.covstr.random[i].subj))")
        println(io, "    Type: $(lmm.covstr.random[i].covtype.s) ($(lmm.covstr.t[i])), Subjects: $(lmm.covstr.sn[i])")
        #println(io, "   Coefnames: $(coefnames(lmm.covstr.schema[i]))")
    end
    println(io, "Repeated: ")
    println(io, "    Model: $(lmm.covstr.repeated.model === nothing ? "nothing" : string(lmm.covstr.repeated.model, "|", lmm.covstr.repeated.subj))")
    println(io, "    Type: $(lmm.covstr.repeated.covtype.s) ($(lmm.covstr.t[end]))")
    println(io, "    Blocks: $(length(lmm.covstr.vcovblock)), Maximum block size: $(maximum(length.(lmm.covstr.vcovblock)))")
    println(io, "")
    if lmm.result.fit
        print(io, "Status: ")
        printresult(io, lmm.result.optim)
        #Optim.converged(lmm.result.optim) ? printstyled(io, "converged \n"; color = :green) : printstyled(io, "not converged \n"; color = :red)
        #if length(lmm.log) > 0  printstyled(io, "Warnings! See lmm.log \n"; color = :yellow) end
        ##println(io, "")
        println(io, "    -2 logREML: ", round(lmm.result.reml, sigdigits = 6), "    BIC: ", round(bic(lmm), sigdigits = 6))
        println(io, "")
        println(io, "    Fixed-effects parameters:")
        #println(io, "")
        #chl = '─'
        z = lmm.result.beta ./ lmm.result.se
        mx  = hcat(coefnames(lmm.mf), round.(lmm.result.beta, sigdigits = 6), round.(lmm.result.se, sigdigits = 6), round.(z, sigdigits = 6), round.(ccdf.(Chisq(1), abs2.(z)), sigdigits = 6))
        mx  = vcat(["Name" "Estimate" "SE" "z" "Pr(>|z|)"], mx)
        printmatrix(io, mx)
        println(io, "")
        println(io, "    Variance components:")
        #println(io, "")
        println(io, "    θ vector: ", round.(lmm.result.theta, sigdigits = 6))
        #println(io, "")

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
    Optim.converged(res) ? printstyled(io, "converged"; color = :green) : printstyled(io, "not converged"; color = :red)
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
