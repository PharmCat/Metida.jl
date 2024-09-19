#lmm.jl

struct LMMLogMsg
    type::Symbol
    msg::String
end


struct ModelStructure
    assign::Vector{Int64}
end

"""
    LMM(model, data; contrasts=Dict{Symbol,Any}(),  random::Union{Nothing, VarEffect, Vector{VarEffect}} = nothing, repeated::Union{Nothing, VarEffect} = nothing, wts::Union{Nothing, AbstractVector, AbstractMatrix, AbstractString, Symbol} = nothing)

Make Linear-Mixed Model object.

`model`: is a fixed-effect model (`@formula`)

`data`: tabular data

`contrasts`: contrasts for fixed factors

`random`: vector of random effects or single random effect

`repeated`: is a repeated effect or vector

`wts`: regression weights (residuals).

Weigts can be set as `Symbol` or `String`, in this case weights taken from tabular data.
If weights is vector then this vector applyed to R-side part of covariance matrix (see [Weights details](@ref weights_header)).
If weights is matrix then R-side part of covariance matrix multiplied by corresponding part of weight-matrix.

See also: [`@lmmformula`](@ref)
"""
struct LMM{T <: AbstractFloat, W <: Union{LMMWts, Nothing}} <: MetidaModel
    model::FormulaTerm
    f::FormulaTerm
    modstr::ModelStructure
    covstr::CovStructure
    data::LMMData{T}
    dv::LMMDataViews{T}
    nfixed::Int
    rankx::Int
    result::ModelResult
    maxvcbl::Int
    wts::Union{Nothing, LMMWts}
    log::Vector{LMMLogMsg}

    function LMM(model::FormulaTerm,
        f::FormulaTerm,
        modstr::ModelStructure,
        covstr::CovStructure,
        data::LMMData{T},
        dv::LMMDataViews{T},
        nfixed::Int,
        rankx::Int,
        result::ModelResult,
        maxvcbl::Int,
        wts::W,
        log::Vector{LMMLogMsg}) where T where W <: Union{LMMWts, Nothing}
        new{T, W}(model, f, modstr, covstr, data, dv, nfixed, rankx, result, maxvcbl, wts, log)
    end
    function LMM(model, data; 
        contrasts=Dict{Symbol,Any}(),  
        random::Union{Nothing, VarEffect, Vector{VarEffect}} = nothing, 
        repeated::Union{Nothing, VarEffect, Vector{VarEffect}} = nothing, 
        wts::Union{Nothing, AbstractVector, AbstractMatrix, AbstractString, Symbol} = nothing)
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
        if !isnothing(wts) && wts isa Union{AbstractString, Symbol}
            if wts isa String wts = Symbol(wts) end
            union!(tv, (wts,))
        end
        ct = Tables.columntable(data)
        if !(tv ⊆ keys(ct)) 
            error("Column(s) ($(setdiff(tv, keys(ct))) not found in table!") 
        end
        data, data_ = StatsModels.missing_omit(NamedTuple{tuple(tv...)}(ct))
        lmmlog = Vector{LMMLogMsg}(undef, 0)
        sch    = schema(model, data, contrasts)
        f      = apply_schema(model, sch, MetidaModel)

        rmf, lmf = modelcols(f, data)

        assign = StatsModels.asgn(f) 

        #mf     = ModelFrame(f, sch, data, MetidaModel)
        #mf     = ModelFrame(model, data; contrasts = contrasts)
        #mm     = ModelMatrix(mf)
        nfixed = fixedeffn(f)
        if repeated === nothing
            repeated = NOREPEAT
        end

        if random === nothing
            random = VarEffect(Metida.@covstr(0|1), Metida.RZero())
        end
        if !isa(random, Vector) random = [random] end
        if !isa(repeated, Vector) repeated = [repeated] end
        for r in repeated
            if r.covtype.s == :SI && !isa(r.model, ConstantTerm)
                lmmlog!(lmmlog, 1, LMMLogMsg(:WARN, "Repeated effect not a constant, but covariance type is SI. "))
            end
        end
        #rmf = response(mf)
        if !(eltype(rmf) <: AbstractFloat) @warn "Response variable not <: AbstractFloat" end 
        lmmdata = LMMData(lmf, rmf)

        covstr = CovStructure(random, repeated, data)
        coefn = size(lmmdata.xv, 2)
        rankx =  rank(lmmdata.xv)
        if rankx != coefn
            @warn "Fixed-effect matrix not full-rank!"
            lmmlog!(lmmlog, 1, LMMLogMsg(:WARN, "Fixed-effect matrix not full-rank!"))
        end

        if isnothing(wts)
            lmmwts = nothing
        else
            if wts isa Symbol
                wts = Tables.getcolumn(data, wts)
            end
            if wts isa AbstractVector
                if length(lmmdata.yv) == length(wts)
                    if any(x -> x <= zero(x), wts) error("Only cases with positive weights allowed!") end
                    lmmwts = LMMWts(wts, covstr.vcovblock)
                else
                    @warn "wts count not equal observations count! wts not used."
                    lmmwts = nothing
                end
            elseif wts isa AbstractMatrix
                if length(lmmdata.yv) == LinearAlgebra.checksquare(wts)
                    if any(x -> x <= zero(x), wts) error("Only positive values allowed!") end
                    lmmwts = LMMWts(wts, covstr.vcovblock)
                else
                    @warn "wts count not equal observations count! wts not used."
                    lmmwts = nothing
                end
            end
        end

        mres = ModelResult(false, nothing, fill(NaN, covstr.tl), NaN, fill(NaN, coefn), nothing, fill(NaN, coefn, coefn), fill(NaN, coefn), nothing, false)

        return LMM(model, f, ModelStructure(assign), covstr, lmmdata, LMMDataViews(lmmdata.xv, lmmdata.yv, covstr.vcovblock), nfixed, rankx, mres, findmax(length, covstr.vcovblock)[1], lmmwts, lmmlog)
    end
    function LMM(f::LMMformula, data; kwargs...)
        return LMM(f.formula, data; random = f.random, repeated = f.repeated, kwargs...)
    end
end

################################################################################
"""
    thetalength(lmm::LMM)

Length of theta vector.
"""
function thetalength(lmm)
    return lmm.covstr.tl
end

"""
    coefn(lmm)

Coef number.
"""
function coefn(lmm)
    return length(lmm.result.beta)
end

"""
    theta(lmm::LMM)

Return theta vector.
"""
function theta(lmm::LMM)
    return copy(theta_(lmm))
end
function theta_(lmm::LMM)
    return lmm.result.theta
end
"""
    rankx(lmm::LMM)

Return rank of `X` matrix.
"""
function rankx(lmm::LMM)
    return Int(lmm.rankx)
end

function nblocks(mm::MetidaModel)
    return length(mm.covstr.vcovblock)
end
function maxblocksize(mm::MetidaModel)
    return mm.maxvcbl
end
function assign(lmm::LMM)
    return lmm.modstr.assign
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
    return lmmlog!(stdout, lmmlog, verbose, vmsg)
end
function lmmlog!(io, lmm::LMM, verbose, vmsg)
    return lmmlog!(io, lmm.log, verbose, vmsg)
end
#MetidaNLopt use this
function lmmlog!(lmm::LMM, verbose, vmsg)
    return lmmlog!(stdout, lmm, verbose, vmsg)
end
function lmmlog!(lmm::LMM, vmsg)
    return lmmlog!(stdout, lmm, 1, vmsg)
end

function msgnum(log::Vector{LMMLogMsg}, type::Symbol)
    n = 0
    for i in log
        if i.type == type
            n += 1
        end
    end
    return n
end
function msgnum(log::Vector{LMMLogMsg})
    return length(log)
end
################################################################################

function Base.show(io::IO, lmm::LMM)
    println(io, "Linear Mixed Model: ", lmm.model)
    rn = lmm.covstr.rn
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
    if lmm.covstr.repeated[1].formula == NOREPEAT.formula
        println(io,"    Residual only")
    else
        for i = 1:length(lmm.covstr.repeated)
            println(io, "    Model: $(lmm.covstr.repeated[i].model === nothing ? "nothing" : string(lmm.covstr.repeated[i].model, "|", lmm.covstr.repeated[i].subj))")
            println(io, "    Type: $(lmm.covstr.repeated[i].covtype.s) ($(lmm.covstr.t[rn+i]))")
        end
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

        mx = hcat(Matrix{Any}(missing, lmm.covstr.tl, 1), lmm.covstr.rcnames, lmm.covstr.ct, round.(lmm.result.theta, sigdigits = 6))

        for i = 1:length(lmm.covstr.random)
            if !isa(lmm.covstr.random[i].covtype.s, ZERO)
                view(mx, lmm.covstr.tr[i], 1) .= "Random $i"
            end
        end
        if length(lmm.covstr.repeated) == 1
            view(mx, lmm.covstr.tr[end], 1) .= "Residual"
        else
            for i = 1:length(lmm.covstr.repeated)
                view(mx, lmm.covstr.tr[lmm.covstr.rn + i], 1) .= "Residual $i"
            end

        end
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
    return Optim.converged(res) ? printstyled(io, "converged"; color = :green) : printstyled(io, "not converged"; color = :red)
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
    return lmm.log
end

################################################################################

function Base.getproperty(x::LMM, s::Symbol)
    if s == :θ
        return x.result.theta
    elseif s == :β
        return x.result.beta
    end
    return getfield(x, s)
end

#=
function Base.convert(::Type{StatsModels.TableRegressionModel}, lmm::LMM)
    StatsModels.TableRegressionModel(lmm, lmm.mf, lmm.mm)
end
=#