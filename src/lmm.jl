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

See also: [`@lmmformula`](@ref)
"""
struct LMM{T<:AbstractFloat} <: MetidaModel
    model::FormulaTerm
    mf::ModelFrame
    mm::ModelMatrix
    covstr::CovStructure{T}
    data::LMMData{T}
    dv::LMMDataViews{T}
    nfixed::Int
    rankx::Int
    result::ModelResult
    maxvcbl::Int
    log::Vector{LMMLogMsg}

    function LMM(model::FormulaTerm,
        mf::ModelFrame,
        mm::ModelMatrix,
        covstr::CovStructure{T},
        data::LMMData{T},
        dv::LMMDataViews{T},
        nfixed::Int,
        rankx::Int,
        result::ModelResult,
        maxvcbl::Int,
        log::Vector{LMMLogMsg}) where T
        new{eltype(mm.m)}(model, mf, mm, covstr, data, dv, nfixed, rankx, result, maxvcbl, log)
    end
    function LMM(model, data; contrasts=Dict{Symbol,Any}(),  random::Union{Nothing, VarEffect, Vector{VarEffect}} = nothing, repeated::Union{Nothing, VarEffect} = nothing)
        #need check responce - Float
        if !Tables.istable(data) error("Data not a table!") end
        if repeated === nothing && random === nothing
            error("No effects specified!")
        end

        tv = termvars(model)
        if !isnothing(random)
            union!(tv, termvars(random))
        end
        if !isnothing(repeated)
            union!(tv, termvars(repeated))
        end
        ct = Tables.columntable(data)
        if !(tv ⊆ keys(ct)) error("Some column(s) not found!") end
        data, data_ = StatsModels.missing_omit(NamedTuple{tuple(tv...)}(ct))
        lmmlog = Vector{LMMLogMsg}(undef, 0)
        sch    = schema(model, data, contrasts)
        f      = apply_schema(model, sch, MetidaModel)
        mf     = ModelFrame(f, sch, data, MetidaModel)
        #mf     = ModelFrame(model, data; contrasts = contrasts)
        mm     = ModelMatrix(mf)
        nfixed = nterms(mf)
        if repeated === nothing
            repeated = NOREPEAT
        end

        if random === nothing
            random = VarEffect(Metida.@covstr(0|1), Metida.RZero())
        end
        if !isa(random, Vector) random = [random] end

        if repeated.covtype.s == :SI && !isa(repeated.model, ConstantTerm)
            lmmlog!(lmmlog, 1, LMMLogMsg(:WARN, "Repeated effect not a constant, but covariance type is SI. "))
        end
        lmmdata = LMMData(modelmatrix(mf), response(mf))
        covstr = CovStructure(random, repeated, data)
        rankx =  rank(lmmdata.xv)
        if rankx != size(lmmdata.xv, 2)
            lmmlog!(lmmlog, 1, LMMLogMsg(:WARN, "Fixed-effect matrix not full-rank!"))
        end
        mres = ModelResult(false, nothing, fill(NaN, covstr.tl), NaN, fill(NaN, rankx), nothing, fill(NaN, rankx, rankx), fill(NaN, rankx), nothing, false)
        LMM(model, mf, mm, covstr, lmmdata, LMMDataViews(lmmdata.xv, lmmdata.yv, covstr.vcovblock), nfixed, rankx, mres, findmax(length, covstr.vcovblock)[1], lmmlog)
    end
    function LMM(f::LMMformula, data; contrasts=Dict{Symbol,Any}(), kwargs...)
        LMM(f.formula, data; contrasts=contrasts,  random = f.random, repeated = f.repeated)
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
    copy(theta_(lmm))
end
function theta_(lmm::LMM)
    lmm.result.theta
end
"""
    rankx(lmm::LMM)

Return rank of `X` matrix.
"""
function rankx(lmm::LMM)
    Int(lmm.rankx)
end

function nblocks(mm::MetidaModel)
    return length(mm.covstr.vcovblock)
end
function maxblocksize(mm::MetidaModel)
    mm.maxvcbl
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

function msgnum(log::Vector{LMMLogMsg}, type::Symbol)
    n = 0
    for i in log
        if i.type == type
            n += 1
        end
    end
    n
end
function msgnum(log::Vector{LMMLogMsg})
    length(log)
end
################################################################################

function Base.show(io::IO, lmm::LMM)
    println(io, "Linear Mixed Model: ", lmm.model)
    for i = 1:length(lmm.covstr.random)
        println(io, "Random $i: ")
        if !lmm.covstr.random[i].covtype.z
            println(io, "   No")
            continue
        end
        println(io, "    Model: $(lmm.covstr.random[i].model === nothing ? "nothing" : string(lmm.covstr.random[i].model, "|", lmm.covstr.random[i].subj))")
        println(io, "    Type: $(lmm.covstr.random[i].covtype.s) ($(lmm.covstr.t[i])), Subjects: $(lmm.covstr.sn[i])")
        
    end
    println(io, "Repeated: ")
    if lmm.covstr.repeated.formula == NOREPEAT.formula
        println(io,"    Residual only")
    else
        println(io, "    Model: $(lmm.covstr.repeated.model === nothing ? "nothing" : string(lmm.covstr.repeated.model, "|", lmm.covstr.repeated.subj))")
        println(io, "    Type: $(lmm.covstr.repeated.covtype.s) ($(lmm.covstr.t[end]))")
    end
    println(io, "Blocks: $(nblocks(lmm)), Maximum block size: $(maxblocksize(lmm))")

    if lmm.result.fit
        print(io, "Status: ")
        printresult(io, lmm.result.optim)
        if any(x-> x.type == :ERROR, lmm.log)
            printstyled(io, "  See error(s) in log. Final results can be wrong!\n"; color = :red)
        elseif any(x-> x.type == :WARN, lmm.log)
            printstyled(io, " See warnings in log.\n"; color = :yellow)
        else
            println(io, " (No Errors)")
        end

        println(io, "    -2 logREML: ", round(lmm.result.reml, sigdigits = 6), "    BIC: ", round(bic(lmm), sigdigits = 6))
        println(io, "")
        println(io, "    Fixed-effects parameters:")

        ct = coeftable(lmm)
        println(io, ct)

        println(io, "    Variance components:")
        println(io, "    θ vector: ", round.(lmm.result.theta, sigdigits = 6))

        mx = hcat(Matrix{Any}(undef, lmm.covstr.tl, 1), lmm.covstr.rcnames, lmm.covstr.ct, round.(lmm.result.theta, sigdigits = 6))

        for i = 1:length(lmm.covstr.random)
            if !isa(lmm.covstr.random[i].covtype.s, ZERO)
                view(mx, lmm.covstr.tr[i], 1) .= "Random $i"
            end
        end
        view(mx, lmm.covstr.tr[end], 1) .= "Residual"
        for i = 1:lmm.covstr.tl
            if mx[i, 3] == :var mx[i, 4] = round.(mx[i, 4]^2, sigdigits = 6) end
        end
        pretty_table(io, mx; show_header = false, alignment=:l, tf = tf_borderless)
    else
        if !any(x-> x.type == :ERROR, lmm.log)
            println(io, "Not fitted.")
        else
            printstyled(io, "Not fitted. See error(s) in log.\n"; color = :red)
        end
    end
end

function printresult(io, res::T) where T <: Optim.MultivariateOptimizationResults
    Optim.converged(res) ? printstyled(io, "converged"; color = :green) : printstyled(io, "not converged"; color = :red)
end
function printresult(io, res)
    if res[3] == :FTOL_REACHED || res[3] == :XTOL_REACHED || res[3] == :SUCCESS
        printstyled(io, "converged ($(string(res[3])))"; color = :green)
    else
        printstyled(io, "not converged ($(string(res[3])))"; color = :red)
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
"""
    getlog(lmm::LMM)

Return fitting log.
"""
function getlog(lmm::LMM)
    lmm.log
end

################################################################################

function Base.getproperty(x::LMM, s::Symbol)
    if s == :θ
        return x.result.theta
    elseif s == :β
        return x.result.beta
    end
    getfield(x, s)
end
