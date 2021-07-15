# Metida
# Copyright © 2019-2020 Vladimir Arnautov aka PharmCat <mail@pharmcat.net>

__precompile__()
module Metida

using Distributions, LinearAlgebra, StatsBase, ForwardDiff, CategoricalArrays, LoopVectorization
using Optim, LineSearches
using MetidaBase

import MetidaBase: Tables, MetidaModel, AbstractCovarianceStructure, AbstractCovmatMethod, AbstractCovarianceType, AbstractLMMDataBlocks, MetidaTable, metida_table, PrettyTables

import LinearAlgebra:checksquare
import StatsModels: @formula
import StatsBase: fit, fit!, coef, coefnames, confint, nobs, dof_residual, dof, loglikelihood, aic, bic, aicc, isfitted, vcov, stderror, modelmatrix, response
import Base:show

export @formula, @covstr,
SI, ScaledIdentity,
DIAG, Diag,
AR, Autoregressive,
ARH, HeterogeneousAutoregressive,
CS, CompoundSymmetry,
CSH, HeterogeneousCompoundSymmetry,
ARMA, AutoregressiveMovingAverage,
TOEP, Toeplitz,
TOEPP, ToeplitzParameterized,
TOEPH, HeterogeneousToeplitz,
TOEPHP, HeterogeneousToeplitzParameterized,
CovarianceType, CovmatMethod,
fit!, LMM, VarEffect, theta, logreml, m2logreml, thetalength, dof_satter, dof_contain, rankx, caic, lcontrast, anova,
gmatrix, rmatrix, vmatrix!,
AbstractCovarianceType, AbstractCovmatMethod, MetidaModel

export coef, coefnames, confint, nobs, dof_residual, dof, loglikelihood, aic, bic, aicc, isfitted, vcov, stderror, modelmatrix, response

include("sweep.jl")
include("varstruct.jl")
include("gmat.jl")
include("rmat.jl")
include("linearalgebra.jl")
include("options.jl")
include("modelresult.jl")
include("lmmdata.jl")
include("lmm.jl")
include("reml.jl")
include("ml.jl")
include("fit.jl")
include("showutils.jl")
include("statsbase.jl")
include("utils.jl")
include("dof_satter.jl")
include("dof_contain.jl")
include("fvalue.jl")
include("anova.jl")

#=
# Use
#    @warnpcfail precompile(args...)
# if you want to be warned when a precompile directive fails
macro warnpcfail(ex::Expr)
    modl = __module__
    file = __source__.file === nothing ? "?" : String(__source__.file)
    line = __source__.line
    quote
        $(esc(ex)) || @warn """precompile directive
     $($(Expr(:quote, ex)))
 failed. Please report an issue in $($modl) (after checking for duplicates) or remove this directive.""" _file=$file _line=$line
    end
end


const __bodyfunction__ = Dict{Method,Any}()

# Find keyword "body functions" (the function that contains the body
# as written by the developer, called after all missing keyword-arguments
# have been assigned values), in a manner that doesn't depend on
# gensymmed names.
# `mnokw` is the method that gets called when you invoke it without
# supplying any keywords.
function __lookup_kwbody__(mnokw::Method)
    function getsym(arg)
        isa(arg, Symbol) && return arg
        @assert isa(arg, GlobalRef)
        return arg.name
    end

    f = get(__bodyfunction__, mnokw, nothing)
    if f === nothing
        fmod = mnokw.module
        # The lowered code for `mnokw` should look like
        #   %1 = mkw(kwvalues..., #self#, args...)
        #        return %1
        # where `mkw` is the name of the "active" keyword body-function.
        ast = Base.uncompressed_ast(mnokw)
        if isa(ast, Core.CodeInfo) && length(ast.code) >= 2
            callexpr = ast.code[end-1]
            if isa(callexpr, Expr) && callexpr.head == :call
                fsym = callexpr.args[1]
                if isa(fsym, Symbol)
                    f = getfield(fmod, fsym)
                elseif isa(fsym, GlobalRef)
                    if fsym.mod === Core && fsym.name === :_apply
                        f = getfield(mnokw.module, getsym(callexpr.args[2]))
                    elseif fsym.mod === Core && fsym.name === :_apply_iterate
                        f = getfield(mnokw.module, getsym(callexpr.args[3]))
                    else
                        f = getfield(fsym.mod, fsym.name)
                    end
                else
                    f = missing
                end
            else
                f = missing
            end
        else
            f = missing
        end
        __bodyfunction__[mnokw] = f
    end
    return f
end

function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(fit!)),NamedTuple{(:aifirst, :init), Tuple{Bool, Vector{Float64}}},typeof(fit!),LMM{Float64}})
    Base.precompile(Tuple{Core.kwftype(typeof(fit!)),NamedTuple{(:aifirst,), Tuple{Bool}},typeof(fit!),LMM{Float64}})
    Base.precompile(Tuple{Core.kwftype(typeof(fit!)),NamedTuple{(:aifirst,), Tuple{Symbol}},typeof(fit!),LMM{Float64}})
    Base.precompile(Tuple{Core.kwftype(typeof(fit!)),NamedTuple{(:hes,), Tuple{Bool}},typeof(fit!),LMM{Float64}})
    Base.precompile(Tuple{Core.kwftype(typeof(fit!)),NamedTuple{(:init,), Tuple{Vector{Float64}}},typeof(fit!),LMM{Float64}})
    Base.precompile(Tuple{Core.kwftype(typeof(fit!)),NamedTuple{(:verbose, :io), Tuple{Int64, IOBuffer}},typeof(fit!),LMM{Float64}})
    Base.precompile(Tuple{Core.kwftype(typeof(printmatrix)),NamedTuple{(:header,), Tuple{Bool}},typeof(printmatrix),IOBuffer,Matrix{Any}})
    Base.precompile(Tuple{Core.kwftype(typeof(varlinkvecapply!)),NamedTuple{(:rholinkf,), Tuple{Symbol}},typeof(varlinkvecapply!),Vector{Float64},Vector{Symbol}})
    Base.precompile(Tuple{Type{CovarianceType},CovmatMethod})
    Base.precompile(Tuple{typeof(TOEPHP),Int64})
    Base.precompile(Tuple{typeof(TOEPP),Int64})
    Base.precompile(Tuple{typeof(aicc),LMM{Float64}})
    Base.precompile(Tuple{typeof(anova),LMM{Float64}})
    Base.precompile(Tuple{typeof(caic),LMM{Float64}})
    Base.precompile(Tuple{typeof(confint),LMM{Float64}})
    Base.precompile(Tuple{typeof(covstrparam),CovarianceType{CovmatMethod},Int64,Int64})
    Base.precompile(Tuple{typeof(dof),LMM{Float64}})
    Base.precompile(Tuple{typeof(dof_contain),LMM{Float64}})
    Base.precompile(Tuple{typeof(dof_satter),LMM{Float64},Int64})
    Base.precompile(Tuple{typeof(dof_satter),LMM{Float64},Matrix{Float64}})
    Base.precompile(Tuple{typeof(dof_satter),LMM{Float64},Matrix{Int64}})
    Base.precompile(Tuple{typeof(dof_satter),LMM{Float64}})
    Base.precompile(Tuple{typeof(fit!),LMM{Float64}})
    Base.precompile(Tuple{typeof(fulldummycodingdict),InteractionTerm{Tuple{Term, Term}}})
    Base.precompile(Tuple{typeof(fulldummycodingdict),Term})
    Base.precompile(Tuple{typeof(fvalue),LMM{Float64},Matrix{Float64}})
    Base.precompile(Tuple{typeof(fvalue),LMM{Float64},Matrix{Int64}})
    Base.precompile(Tuple{typeof(gmat_ar!),Matrix{Float64},Vector{Float64},Int64})
    Base.precompile(Tuple{typeof(gmat_arh!),Matrix{Float64},Vector{Float64},Int64})
    Base.precompile(Tuple{typeof(gmat_arma!),Matrix{Float64},Vector{Float64},Int64})
    Base.precompile(Tuple{typeof(gmat_cs!),Matrix{Float64},Vector{Float64},Int64})
    Base.precompile(Tuple{typeof(gmat_csh!),Matrix{Float64},Vector{Float64},CovmatMethod})
    Base.precompile(Tuple{typeof(gmat_csh!),Matrix{Float64},Vector{Float64},Int64})
    Base.precompile(Tuple{typeof(gmat_diag!),Matrix{Float64},Vector{Float64},Int64})
    Base.precompile(Tuple{typeof(gmat_si!),Matrix{Float64},Vector{Float64},Int64})
    Base.precompile(Tuple{typeof(gmat_toephp!),Matrix{Float64},Vector{Float64},Int64})
    Base.precompile(Tuple{typeof(gmat_toepp!),Matrix{Float64},Vector{Float64},Int64})
    Base.precompile(Tuple{typeof(gradc),LMM{Float64},Vector{Float64}})
    Base.precompile(Tuple{typeof(hessian),LMM{Float64},Vector{Float64}})
    Base.precompile(Tuple{typeof(logreml),LMM{Float64}})
    Base.precompile(Tuple{typeof(modelmatrix),LMM{Float64}})
    Base.precompile(Tuple{typeof(mulθ₃),SubArray{Float64, 1, Vector{Float64}, Tuple{Vector{UInt32}}, false},SubArray{Float64, 2, Matrix{Float64}, Tuple{Vector{UInt32}, Base.Slice{Base.OneTo{Int64}}}, false},Vector{Float64},SubArray{Float64, 2, Matrix{Float64}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false}})
    Base.precompile(Tuple{typeof(nterms),ModelFrame{NamedTuple{(:expresp, :factor, :exptime), Tuple{Vector{Float64}, Vector{String}, Vector{Float64}}}, StatisticalModel}})
    Base.precompile(Tuple{typeof(nterms),ModelFrame{NamedTuple{(:lnpk, :sequence, :period, :treatment), Tuple{Vector{Float64}, Vector{String}, Vector{String}, Vector{String}}}, StatisticalModel}})
    Base.precompile(Tuple{typeof(nterms),ModelFrame{NamedTuple{(:var, :sequence, :period, :formulation), Tuple{Vector{Float64}, Vector{String}, Vector{String}, Vector{String}}}, StatisticalModel}})
    Base.precompile(Tuple{typeof(printmatrix),IOBuffer,Matrix{Any}})
    Base.precompile(Tuple{typeof(printresult),IOBuffer,Optim.MultivariateOptimizationResults{Newton{InitialHagerZhang{Float64}, HagerZhang{Float64, Base.RefValue{Bool}}}, Float64, Vector{Float64}, Float64, Float64, Vector{OptimizationState{Float64, Newton{InitialHagerZhang{Float64}, HagerZhang{Float64, Base.RefValue{Bool}}}}}, Bool, NamedTuple{(:f_limit_reached, :g_limit_reached, :h_limit_reached, :time_limit, :callback, :f_increased), NTuple{6, Bool}}}})
    Base.precompile(Tuple{typeof(response),LMM{Float64}})
    Base.precompile(Tuple{typeof(rmat_base_inc!),Matrix{Float64},Vector{Float64},CovStructure{Float64},Vector{UInt32},Vector{Vector{Vector{UInt32}}}})
    Base.precompile(Tuple{typeof(rmatp_arh!),SubArray{Float64, 2, Matrix{Float64}, Tuple{Vector{Int64}, Vector{Int64}}, false},Vector{Float64},SubArray{Float64, 2, Matrix{Float64}, Tuple{Vector{UInt32}, Base.Slice{Base.OneTo{Int64}}}, false},Int64})
    Base.precompile(Tuple{typeof(rmatp_arma!),SubArray{Float64, 2, Matrix{Float64}, Tuple{Vector{Int64}, Vector{Int64}}, false},Vector{Float64},SubArray{Float64, 2, Matrix{Float64}, Tuple{Vector{UInt32}, Base.Slice{Base.OneTo{Int64}}}, false},Int64})
    Base.precompile(Tuple{typeof(rmatp_cs!),SubArray{Float64, 2, Matrix{Float64}, Tuple{Vector{Int64}, Vector{Int64}}, false},Vector{Float64},SubArray{Float64, 2, Matrix{Float64}, Tuple{Vector{UInt32}, Base.Slice{Base.OneTo{Int64}}}, false},Int64})
    Base.precompile(Tuple{typeof(rmatp_csh!),SubArray{Float64, 2, Matrix{Float64}, Tuple{Vector{Int64}, Vector{Int64}}, false},Vector{Float64},SubArray{Float64, 2, Matrix{Float64}, Tuple{Vector{UInt32}, Base.Slice{Base.OneTo{Int64}}}, false},Int64})
    Base.precompile(Tuple{typeof(rmatp_diag!),SubArray{Float64, 2, Matrix{Float64}, Tuple{Vector{Int64}, Vector{Int64}}, false},Vector{Float64},SubArray{Float64, 2, Matrix{Float64}, Tuple{Vector{UInt32}, Base.Slice{Base.OneTo{Int64}}}, false},CovmatMethod})
    Base.precompile(Tuple{typeof(rmatp_si!),SubArray{Float64, 2, Matrix{Float64}, Tuple{Vector{UInt32}, Vector{UInt32}}, false},Vector{Float64},SubArray{Float64, 2, Matrix{Float64}, Tuple{Vector{UInt32}, Base.Slice{Base.OneTo{Int64}}}, false},Int64})
    Base.precompile(Tuple{typeof(rmatp_spexp!),SubArray{Float64, 2, Matrix{Float64}, Tuple{Vector{Int64}, Vector{Int64}}, false},Vector{Float64},SubArray{Float64, 2, Matrix{Float64}, Tuple{Vector{UInt32}, Base.Slice{Base.OneTo{Int64}}}, false},Int64})
    Base.precompile(Tuple{typeof(rmatp_toephp!),SubArray{Float64, 2, Matrix{Float64}, Tuple{Vector{Int64}, Vector{Int64}}, false},Vector{Float64},SubArray{Float64, 2, Matrix{Float64}, Tuple{Vector{UInt32}, Base.Slice{Base.OneTo{Int64}}}, false},Int64})
    Base.precompile(Tuple{typeof(rmatp_toepp!),SubArray{Float64, 2, Matrix{Float64}, Tuple{Vector{Int64}, Vector{Int64}}, false},Vector{Float64},SubArray{Float64, 2, Matrix{Float64}, Tuple{Vector{UInt32}, Base.Slice{Base.OneTo{Int64}}}, false},Int64})
    Base.precompile(Tuple{typeof(rmatrix),LMM{Float64},Int64})
    Base.precompile(Tuple{typeof(show),IOBuffer,ANOVATable})
    Base.precompile(Tuple{typeof(show),IOBuffer,CovStructure{Float64}})
    Base.precompile(Tuple{typeof(show),IOBuffer,LMM{Float64}})
    Base.precompile(Tuple{typeof(show),IOBuffer,ModelResult})
    Base.precompile(Tuple{typeof(updatenametype!),Vector{Symbol},Vector{String},Tuple{Int64, Int64, Int64},Tuple{ContinuousTerm{Float64}, ContinuousTerm{Float64}},Symbol})
    Base.precompile(Tuple{typeof(updatenametype!),Vector{Symbol},Vector{String},Tuple{Int64, Int64},CategoricalTerm{StatsModels.FullDummyCoding, String, 2},Symbol})
    Base.precompile(Tuple{typeof(updatenametype!),Vector{Symbol},Vector{String},Tuple{Int64, Int64},CategoricalTerm{StatsModels.FullDummyCoding, String, 3},Symbol})
    Base.precompile(Tuple{typeof(updatenametype!),Vector{Symbol},Vector{String},Tuple{Int64, Int64},CategoricalTerm{StatsModels.FullDummyCoding, String, 4},Symbol})
    Base.precompile(Tuple{typeof(updatenametype!),Vector{Symbol},Vector{String},Tuple{Int64, Int64},CategoricalTerm{StatsModels.FullDummyCoding, String, 5},Symbol})
    Base.precompile(Tuple{typeof(updatenametype!),Vector{Symbol},Vector{String},Tuple{Int64, Int64},CategoricalTerm{StatsModels.FullDummyCoding, String, 7},Symbol})
    Base.precompile(Tuple{typeof(updatenametype!),Vector{Symbol},Vector{String},Tuple{Int64, Int64},ContinuousTerm{Float64},Symbol})
    Base.precompile(Tuple{typeof(updatenametype!),Vector{Symbol},Vector{String},Tuple{Int64, Int64},InteractionTerm{Tuple{CategoricalTerm{StatsModels.FullDummyCoding, String, 3}, CategoricalTerm{StatsModels.FullDummyCoding, String, 2}}},Symbol})
    Base.precompile(Tuple{typeof(updatenametype!),Vector{Symbol},Vector{String},Tuple{Int64, Int64},InterceptTerm{false},Symbol})
    Base.precompile(Tuple{typeof(updatenametype!),Vector{Symbol},Vector{String},Tuple{Int64, Int64},InterceptTerm{true},Symbol})
    Base.precompile(Tuple{typeof(updatenametype!),Vector{Symbol},Vector{String},Tuple{Int64, Int64},Tuple{InterceptTerm{true}, CategoricalTerm{DummyCoding, String, 1}, CategoricalTerm{DummyCoding, String, 2}, InteractionTerm{Tuple{CategoricalTerm{DummyCoding, String, 1}, CategoricalTerm{DummyCoding, String, 2}}}},Symbol})
    Base.precompile(Tuple{typeof(updatenametype!),Vector{Symbol},Vector{String},Tuple{Int64, Int64},Tuple{InterceptTerm{true}, CategoricalTerm{DummyCoding, String, 1}},Symbol})
    Base.precompile(Tuple{typeof(updatenametype!),Vector{Symbol},Vector{String},Tuple{Int64, Int64},Tuple{InterceptTerm{true}, CategoricalTerm{DummyCoding, String, 2}, CategoricalTerm{DummyCoding, String, 1}},Symbol})
    Base.precompile(Tuple{typeof(updatenametype!),Vector{Symbol},Vector{String},Tuple{Int64, Int64},Tuple{InterceptTerm{true}, ContinuousTerm{Float64}},Symbol})
    Base.precompile(Tuple{typeof(vmatrix),LMM{Float64},Int64})
    Base.precompile(Tuple{typeof(vmatrix),Vector{Float64},LMM{Float64},Int64})
    isdefined(Metida, Symbol("#hfunc!#33")) && Base.precompile(Tuple{getfield(Metida, Symbol("#hfunc!#33")),Matrix{Float64},Vector{Float64}})
    let fbody = try __lookup_kwbody__(which(fit!, (LMM{Float64},))) catch missing end
        if !ismissing(fbody)
            precompile(fbody, (Symbol,Symbol,Symbol,Symbol,Bool,Float64,Float64,Float64,Bool,Nothing,Base.TTY,Int64,Int64,typeof(fit!),LMM{Float64},))
        end
    end
    let fbody = try __lookup_kwbody__(which(fit!, (LMM{Float64},))) catch missing end
        if !ismissing(fbody)
            precompile(fbody, (Symbol,Symbol,Symbol,Symbol,Bool,Float64,Float64,Float64,Bool,Vector{Float64},Base.TTY,Int64,Int64,typeof(fit!),LMM{Float64},))
        end
    end
    let fbody = try __lookup_kwbody__(which(fit!, (LMM{Float64},))) catch missing end
        if !ismissing(fbody)
            precompile(fbody, (Symbol,Symbol,Symbol,Symbol,Symbol,Float64,Float64,Float64,Bool,Nothing,Base.TTY,Int64,Int64,typeof(fit!),LMM{Float64},))
        end
    end
    let fbody = try __lookup_kwbody__(which(fit!, (LMM{Float64},))) catch missing end
        if !ismissing(fbody)
            precompile(fbody, (Symbol,Symbol,Symbol,Symbol,Symbol,Float64,Float64,Float64,Bool,Vector{Float64},Base.TTY,Int64,Int64,typeof(fit!),LMM{Float64},))
        end
    end
end

_precompile_()
=#
#include(".jl")
end # module
