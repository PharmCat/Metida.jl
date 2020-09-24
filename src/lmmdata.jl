

struct LMMData{T}
    xv::Vector{Matrix{T}}
    zv::Vector{Matrix{T}}
    zrv::Vector{Matrix{T}}
    yv::Vector{Vector{T}}
    function LMMData(xa::Vector{Matrix{T}}, za::Vector{Matrix{T}}, rza::Vector{Matrix{T}}, ya::Vector{Vector{T}}) where T
        new{T}(xa, za, rza, ya)
    end
end
