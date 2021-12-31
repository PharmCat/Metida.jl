#lmmdata.jl

struct LMMData{T}
    # Fixed effect matrix
    xv::Matrix{T}
    # Responce vector
    yv::Vector{T}
    function LMMData(xa::Matrix{T}, ya::Vector{T}) where T
        new{T}(xa, ya)
    end
end

struct LMMDataViews{T} <: AbstractLMMDataBlocks
    # Fixed effect matrix views
    xv::Vector{Matrix{T}}
    # Responce vector views
    yv::Vector{Vector{T}}
    function LMMDataViews(xv::Vector{Matrix{T}}, yv::Vector{Vector{T}}) where T
        new{T}(xv, yv)
    end
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
        LMMDataViews(x, y)
    end
    function LMMDataViews(lmm::MetidaModel)
        return LMMDataViews(lmm.data.xv, lmm.data.yv, lmm.covstr.vcovblock)
    end
end
