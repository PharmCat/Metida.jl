"""
    Make X, Z matrices and vector y for each subject;
"""
function subjblocks(df, sbj::Symbol, x::Matrix{T1}, z::Matrix{T2}, y::Vector{T3}, rz::T4) where {T1, T2, T3, T4 <: AbstractMatrix}
    u = unique(df[!, sbj])
    #=
    xa  = Vector{SubArray{T1, 2, Matrix{T1}, Tuple{Array{Int64,1}, Base.Slice{Base.OneTo{Int64}}},false}}(undef, length(u))
    za  = Vector{SubArray{T2, 2, Matrix{T2}, Tuple{Array{Int64,1}, Base.Slice{Base.OneTo{Int64}}},false}}(undef, length(u))
    ya  = Vector{SubArray{T3, 1, Vector{T3}, Tuple{Array{Int64,1}},false}}(undef, length(u))
    rza = Vector{SubArray{T4, 2, Matrix{T4}, Tuple{Array{Int64,1}, Base.Slice{Base.OneTo{Int64}}},false}}(undef, length(u))
    =#
    xa  = Vector{AbstractArray}(undef, length(u))
    za  = Vector{AbstractArray}(undef, length(u))
    ya  = Vector{AbstractArray}(undef, length(u))
    rza = Vector{AbstractArray}(undef, length(u))
    @inbounds @simd for i = 1:length(u)
        v = findall(x->x==u[i], df[!, sbj])
        xa[i] = view(x, v, :)
        za[i] = view(z, v, :)
        ya[i] = view(y, v)
        rza[i] = view(rz, v, :)
    end
    return xa, za, rza, ya
end
function subjblocks(df, sbj::Symbol, x::Matrix{T1}, z::Matrix{T2}, y::Vector{T3}, rz::Nothing) where {T1, T2, T3}
    u = unique(df[!, sbj])
    xa  = Vector{SubArray{T1, 2, Matrix{T1}, Tuple{Array{Int64,1}, Base.Slice{Base.OneTo{Int64}}},false}}(undef, length(u))
    za  = Vector{SubArray{T2, 2, Matrix{T2}, Tuple{Array{Int64,1}, Base.Slice{Base.OneTo{Int64}}},false}}(undef, length(u))
    ya  = Vector{SubArray{T3, 1, Vector{T3}, Tuple{Array{Int64,1}},false}}(undef, length(u))
    rza = Vector{Int}(undef, length(u))
    @inbounds @simd for i = 1:length(u)
        v = findall(x->x==u[i], df[!, sbj])
        xa[i]  = view(x, v, :)
        za[i]  = view(z, v, :)
        ya[i]  = view(y, v)
        rza[i] = length(ya[i])
    end
    return xa, za, rza, ya
end
