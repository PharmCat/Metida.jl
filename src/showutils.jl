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

function rcoefnames(s, t, ::Val)
    v = Vector{String}(undef, t)
    v .= "─"
end
function rcoefnames(s, t, ::Val{:CSH})
    cn = coefnames(s)
    if isa(cn, Vector)
        l  = length(cn)
    else
        l  = 1
    end
    v  = Vector{String}(undef, t)
    view(v, 1:l) .= string.(cn)
    v[end] = "Rho"
    v
end
function rcoefnames(s, t, ::Val{:SI})
    ["Var"]
end
function rcoefnames(s, t, ::Val{:VC})
    string.(coefnames(s))
end
