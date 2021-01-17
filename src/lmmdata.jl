

struct LMMData{T}
    # Fixed effect matrix
    xv::Matrix{T}
    # Responce vector
    yv::Vector{T}
    # Global variance blocking factor (subject)
    block::Vector{Vector{Int}}
    # Block factor
    subject::Vector{Symbol}
    function LMMData(xa::Matrix{T}, ya::Vector{T}, block, subject) where T
        new{T}(xa, ya, block, subject)
    end
end
