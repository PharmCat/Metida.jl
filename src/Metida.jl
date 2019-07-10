# Metida
# Copyright © 2019 Vladimir Arnautov aka PharmCat <mail@pharmcat.net>

module Metida

using StatsBase, Statistics, Distributions, DataFrames, QuadGK, Roots, Random
import SpecialFunctions.lgamma

abstract type AbstractMetidaObject end

abstract type AbstractMetidaDataSet <: AbstractMetidaObject end

abstract type AbstractMetidaData <: AbstractMetidaObject end

abstract type AbstractMetidaParam <: AbstractMetidaData end


"""
Proportion type:
    - x
    - y
"""
struct Proportion <: AbstractMetidaParam
    x::Int
    n::Int
end

"""
Numeric variable representation type:
    - mean
    - var
    - n
"""
struct Numeric <: AbstractMetidaParam
    mean::Real
    var::Real
    n::Int
end

"""
Confidence interval type:
"""
struct ConfidenceInterval <: AbstractMetidaObject
    lower::Real
    upper::Real
    estimate::Real
end

"""
Confidence interval estimation result type:
"""
struct CofidenceIntrvalEstimate{T <: Union{AbstractMetidaParam, Tuple{AbstractMetidaParam, AbstractMetidaParam}}} <: AbstractMetidaObject
    param::T
    type::Symbol
    method::Symbol
    ci::ConfidenceInterval
end

# ----

"""
Pharmacokinetics profile type:
"""
struct PKProfile <: AbstractMetidaData
    variable::Symbol
    varname::String
    sort::Tuple{Vararg{Symbol}}
    time::Tuple{Vararg{Real}}
    conc::Tuple{Vararg{Real}}
    dose::Real
    elstart::Int
    elend::Int
end

"""
Pharmacokinetics parameter type:
"""
struct PKParam <: AbstractMetidaData
    variable::Symbol
    varname::String
    sort::Tuple{Vararg{Symbol}}
    cmax::Real
    auc::Real
end

"""
Descriptive statistics type:
"""
struct Descriptive <: AbstractMetidaData
    variable::Symbol
    varname::String
    sort::Tuple{Vararg{Symbol}}
    stat::NamedTuple{}

end

"""
Frequency statistics type
"""
struct Frequency <: AbstractMetidaData
    variable::Symbol
    varname::String
    sort::Tuple{Vararg{Symbol}}
    size::Int
    stat::NamedTuple{}
end

#----
struct MetidaDataSet <: AbstractMetidaDataSet
    data::Tuple{Vararg{Any}}
end

"""
Pharmacokinetics dataset type:
"""
struct PKDataSet <: AbstractMetidaDataSet
    varlist::Tuple{Vararg{Symbol}}
    sortlist::Tuple{Vararg{Any}}
    profiles::Tuple{Vararg{PKProfile}}
    data::Tuple{Vararg{PKParam}}
    log::String
end

"""
Frequency statistics dataset type
"""
struct FrequencyDataSet <: AbstractMetidaDataSet
    varlist::Tuple{Vararg{Symbol}}
    sortlist::Tuple{Vararg{Symbol}}
    data::Tuple{Vararg{Frequency}}
end

"""
Descriptive statistics dataset type:
"""
struct DescriptiveDataSet <: AbstractMetidaDataSet
    varlist::Tuple{Vararg{Symbol}}
    sortlist::Tuple{Vararg{Symbol}}
    data::Tuple{Vararg{Descriptive}}
end

#----

"""
Sample size estimation type:
"""
struct SampleSize <: AbstractMetidaObject
    param::Symbol              #параметр
    type::Symbol               #тип NI/Sup/NonInf/BE
    gnum::Int                  #число групп
    val::Tuple{Vararg{Real}}        #Значения параметров
    var::Tuple{Vararg{Real}}         #дисперсия
    k::Real                    #коэффициент групп
    diff::Real                 #разность
    power::Real                #мощность
    result::Tuple{Vararg{Real}}      #результат
end

"""
Power estimation type:
"""
struct Power <: AbstractMetidaObject
    param::Symbol
    type::Symbol
    gnum::Int
    val::Tuple{Vararg{Real}}
    var::Tuple{Vararg{Real}}
    k::Real
    diff::Real
    power::Real
    result::Tuple{Vararg{Real}}
end

struct BioequvalenceTrial <: AbstractMetidaObject
    design::Symbol
end

struct ContingencyTable <: AbstractMetidaObject
    rown::Symbol
    coln::Symbol
    table::Array{Int, 2}
end

struct ContingencyTableDataSet <: AbstractMetidaDataSet
    varlist::Tuple{Vararg{Symbol}}
    sortlist::Tuple{Vararg{Symbol}}
    data::Tuple{Vararg{ContingencyTable}}
end


# ----

function StatsBase.confint(d::AbstractMetidaData, alpha::Float64=0.05;)

end

function StatsBase.summarystats(d::AbstractMetidaDataSet)

end

function StatsBase.describe(d::AbstractMetidaDataSet)

end

function StatsBase.r2(p::PKParam)

end

function StatsBase.r²(p::PKParam)
    return StatsBase.r2(p)
end

function descriptives() end
function frequency() end
function cmh() end
function samplesize() end
function samplesizesim() end
function power() end
function powersim() end
function nca() end
function ncamodel() end
function htmlexport() end
function cvfromci() end
function pooledcv() end

# ----

# Base.show

function Base.show(io::IO, ::MIME"text/plain", obj::ConfidenceInterval)
    compact = get(io, :compact, false)
    println(io, "Confidence interval:")
    if compact
       print(io, "Lower: $(obj.lower), Upper: $(obj.upper), Estimate: $(obj.estimate)")
    else
       println(io, "Upper: ", obj.upper)
       println(io, "Lower: ", obj.lower)
       print(io, "Estimate: ", obj.estimate)
   end
end

#Docs.getdoc(obj::ConfidenceInterval) = "ConfidenceInterval $(obj.lower)"
#include("descriptive.jl")
#include("ci.jl")

end # module
