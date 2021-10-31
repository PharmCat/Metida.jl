#lmmdata.jl

struct LMMData{T <: AbstractFloat}
    # Fixed effect matrix
    xv::Matrix{T}
    # Responce vector
    yv::Vector{T}
    function LMMData(xa::Matrix{T}, ya::Vector{T}) where T <: AbstractFloat
        new{T}(xa, ya)
    end
end

struct LMMDataViews{T} <: AbstractLMMDataBlocks
    # Fixed effect matrix views
    xv::Vector{Matrix{T}}
    # Responce vector views
    yv::Vector{Vector{T}}
    function LMMDataViews(xv::Matrix{T}, yv::Vector{T}, vcovblock) where T
        #x1 = view(xv, vcovblock[1],:)
        #y1 = view(yv, vcovblock[1])
        x = Vector{Matrix{T}}(undef, length(vcovblock))
        y = Vector{Vector{T}}(undef, length(vcovblock))
        #x[1] = x1
        #y[1] = y1
        for i = 1:length(vcovblock)
            x[i] = xv[vcovblock[i],:]
            y[i] = yv[vcovblock[i]]
        end
        new{T}(x, y)
    end
    function LMMDataViews(lmm)
        return LMMDataViews(lmm.data.xv, lmm.data.yv, lmm.covstr.vcovblock)
    end
end
