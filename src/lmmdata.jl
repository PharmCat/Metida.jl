

struct LMMData{T}
    # Fixed effect matrix
    xv::Matrix{T}
    # Responce vector
    yv::Vector{T}
    function LMMData(xa::Matrix{T}, ya::Vector{T}) where T
        new{T}(xa, ya)
    end
end
