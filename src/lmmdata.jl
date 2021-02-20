

struct LMMData{T <: AbstractFloat}
    # Fixed effect matrix
    xv::Matrix{T}
    # Responce vector
    yv::Vector{T}
    function LMMData(xa::Matrix{T}, ya::Vector{T}) where T <: AbstractFloat
        new{T}(xa, ya)
    end
end

struct LMMDataViews{T1, T2}
    # Fixed effect matrix views
    xv::T1
    # Responce vector views
    yv::T2
    function LMMDataViews(xv, yv, vcovblock)
        x1 = view(xv, vcovblock[1],:)
        y1 = view(yv, vcovblock[1])
        x = Vector{typeof(x1)}(undef, length(vcovblock))
        y = Vector{typeof(y1)}(undef, length(vcovblock))
        x[1] = x1
        y[1] = y1
        for i = 2:length(vcovblock)
            x[i] = view(xv, vcovblock[i],:)
            y[i] = view(yv, vcovblock[i])
        end
        new{typeof(x), typeof(y)}(x, y)
    end
    function LMMDataViews(lmm)
        return LMMDataViews(lmm.data.xv, lmm.data.yv, lmm.covstr.vcovblock)
    end
end
