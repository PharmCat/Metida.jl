"""
    Make X, Z matrices and vector y for each subject;
"""
function subjblocks(df, sbj::Symbol, X::Matrix, Z::Matrix, y::Vector)
    u = unique(df[!, sbj])
    Xa = Vector{AbstractMatrix}(undef, length(u))
    Za = Vector{AbstractMatrix}(undef, length(u))
    ya = Vector{AbstractVector}(undef, length(u))
    @inbounds @simd for i = 1:length(u)
        v = findall(x->x==u[i], df[!, sbj])
        Xa[i] = view(X, v, :)
        Za[i] = view(Z, v, :)
        ya[i] = view(y, v)
    end
    return Xa, Za, ya
end
