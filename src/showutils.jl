#showutils.jl
tf_line = TextFormat(up_right_corner     = '⎯',
                       up_left_corner      = '⎯',
                       bottom_left_corner  = '⎯',
                       bottom_right_corner = '⎯',
                       up_intersection     = '⎯',
                       left_intersection   = '⎯',
                       right_intersection  = '⎯',
                       middle_intersection = '⎯',
                       bottom_intersection = '⎯',
                       column              = ' ',
                       row                 = '⎯',
                       hlines              = [:begin, :header]);

#=
function addspace(s::String, n::Int; first = false)::String
    if n > 0
        for i = 1:n
            if first s = Char(' ') * s else s = s * Char(' ') end
        end
    end
    return s
end

function printmatrix(io::IO, m::Matrix; header = false)
    sm = string.(m)
    lv = maximum(length.(sm), dims = 1)
    for r = 1:size(sm, 1)
        for c = 1:size(sm, 2)
            sm[r,c] = addspace(sm[r,c], lv[c] - length(sm[r,c]))*"   "
        end
    end
    i = 1
    if header
        line = "⎯" ^ sum(length.(sm[1,:]))
        println(io, line)
        for c = 1:size(sm, 2)
            print(io, sm[1,c])
        end
        print(io, "\n")
        println(io, line)
        i = 2
    end
    for r = i:size(sm, 1)
        for c = 1:size(sm, 2)
            print(io, sm[r,c])
        end
        print(io, "\n")
    end
end
=#
