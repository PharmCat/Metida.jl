#lmmdata.jl

struct LMMData{T<:AbstractFloat}
    # Fixed effect matrix
    xv::Matrix{Float64}
    # Responce vector
    yv::Vector{T}
    function LMMData(xa::AbstractMatrix{Float64}, ya::AbstractVector{T}) where T
        new{T}(xa, ya)
    end
    function LMMData(xa::AbstractMatrix{Float64}, ya::AbstractVector{Int})
        LMMData(xa, float.(ya))
    end
end

struct LMMDataViews{T<:AbstractFloat} <: AbstractLMMDataBlocks
    # Fixed effect matrix views
    xv::Vector{Matrix{Float64}}
    # Responce vector views
    yv::Vector{Vector{T}}
    function LMMDataViews(xv::Vector{Matrix{Float64}}, yv::Vector{Vector{T}}) where T
        new{T}(xv, yv)
    end
    function LMMDataViews(xv::Matrix{Float64}, yv::Vector{T}, vcovblock) where T
        x = Vector{Matrix{Float64}}(undef, length(vcovblock))
        y = Vector{Vector{T}}(undef, length(vcovblock))
        for i = 1:length(vcovblock)
            x[i] = xv[vcovblock[i],:]
            y[i] = yv[vcovblock[i]]
        end
        LMMDataViews(x, y)
    end
    function LMMDataViews(lmm::MetidaModel)
        return LMMDataViews(lmm.data.xv, lmm.data.yv, lmm.covstr.vcovblock)
    end
end

struct LMMWts{T<:AbstractFloat} 
    sqrtwts::Vector{Vector{T}}
    function LMMWts(sqrtwts::Vector{Vector{T}}) where T
        new{T}(sqrtwts)
    end
    function LMMWts(wts::Vector{T}, vcovblock) where T
        sqrtwts = Vector{Vector{T}}(undef, length(vcovblock))
        for i in eachindex(vcovblock)
            sqrtwts[i] = @. inv(sqrt($(view(wts, vcovblock[i]))))
        end
        LMMWts(sqrtwts)
    end
end