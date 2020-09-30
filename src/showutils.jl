#showutils.jl

function addspace(s::String, n::Int; first = false)::String
    if n > 0
        for i = 1:n
            if first s = Char(' ') * s else s = s * Char(' ') end
        end
    end
    return s
end
function makepmatrix(m::Matrix)
    sm = string.(m)
    lv = maximum(length.(sm), dims = 1)
    for r = 1:size(sm, 1)
        for c = 1:size(sm, 2)
            sm[r,c] = addspace(sm[r,c], lv[c] - length(sm[r,c]))*"   "
        end
    end
end
function printmatrix(io::IO, m::Matrix)
    sm = string.(m)
    lv = maximum(length.(sm), dims = 1)
    for r = 1:size(sm, 1)
        for c = 1:size(sm, 2)
            print(io, addspace(sm[r,c], lv[c] - length(sm[r,c]))*"   ")
        end
        println(io, "")
    end
end

function rcoefnames(s, t, ve::VarEffect{T}) where T <: AbstractCovarianceType
    v = Vector{String}(undef, t)
    v .= "─"
end
function rcoefnames(s, t, ve::VarEffect{T}) where T <: HeterogeneousCompoundSymmetry
    cn = coefnames(s)
    l  = length(cn)
    v  = Vector{String}(undef, t)
    view(v, 1:l) .= string.(cn)
    v[end] = "Rho"
    v
end
function rcoefnames(s, t, ve::VarEffect{T}) where T <: ScaledIdentity
    ["Var"]
end
function rcoefnames(s, t, ve::VarEffect{T}) where T <: VarianceComponents
    string.(coefnames(s))
end
