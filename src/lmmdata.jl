

struct LMMData{T}
    # Fixed effect matrix
    xv::Matrix{T}
    # Responce vector
    yv::Vector{T}
    # Global variance blocking factor (subject)
    block::Vector{Vector{UInt32}}
    function LMMData(xa::Matrix{T}, ya::Vector{T}, block) where T
        new{T}(xa, ya, block)
    end
end
