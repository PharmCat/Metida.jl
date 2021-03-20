#showutils.jl

function addspace(s::String, n::Int; first = false)::String
    if n > 0
        for i = 1:n
            if first s = Char(' ') * s else s = s * Char(' ') end
        end
    end
    return s
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
