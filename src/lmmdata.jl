

struct LMMData{T}
    xv::Matrix{T}
    yv::Vector{T}
    block::Vector{Vector{Int}}
    function LMMData(xa::Matrix{T}, ya::Vector{T}, block) where T
        new{T}(xa, ya, block)
    end
end
