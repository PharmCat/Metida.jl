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

function rcoefnames(s, t, ve)
    if ve == :SI
        return ["σ² "]
    elseif ve == :DIAG
        return fill!(Vector{String}(undef, length(coefnames(s))), "σ² ") .* string.(coefnames(s))
    elseif ve == :CS || ve == :AR
        return ["σ² ", "ρ "]
    elseif ve == :CSH || ve == :ARH
        cn = coefnames(s)
        if isa(cn, Vector)
            l  = length(cn)
        else
            l  = 1
        end
        v  = Vector{String}(undef, t)
        view(v, 1:l) .= (fill!(Vector{String}(undef, l), "σ² ") .*string.(cn))
        v[end] = "ρ "
        return v
    elseif ve == :ARMA
        return ["σ² ", "γ ", "ρ "]
    elseif ve == :TOEP || ve == :TOEPP
        v = Vector{String}(undef, t)
        v[1] = "σ² "
        if length(v) > 1
            for i = 2:length(v)
                v[i] = "ρ band $(i-1) "
            end
        end
        return v
    elseif ve == :TOEPH || ve == :TOEPHP
        cn = coefnames(s)
        if isa(cn, Vector)
            l  = length(cn)
        else
            l  = 1
        end
        v  = Vector{String}(undef, t)
        view(v, 1:l) .= (fill!(Vector{String}(undef, l), "σ² ") .*string.(cn))
        if length(v) > l
            for i = l+1:length(v)
                v[i] = "ρ band $(i-l) "
            end
        end
        return v
    else
        v = Vector{String}(undef, t)
        v .= "NA"
        return v
    end
end
