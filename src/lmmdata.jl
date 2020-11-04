

struct LMMData{T}
    xv::Matrix{T}
    zv::Matrix{T}
    zrv::Matrix{T}
    yv::Vector{T}
    block::Vector{Vector{Int}}
    function LMMData(xa::Matrix{T}, za::Matrix{T}, rza::Matrix{T}, ya::Vector{T}, block) where T
        new{T}(xa, za, rza, ya, block)
    end
end
