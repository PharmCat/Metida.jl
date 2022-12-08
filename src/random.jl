
"""
    rand(rng::AbstractRNG, lmm::LMM{T}) where T

Generate random responce vector for fitted 'lmm' model.
"""
function rand(rng::AbstractRNG, lmm::LMM)
    if !lmm.result.fit error("Model not fitted!") end
    rand(rng, lmm, lmm.result.theta, lmm.result.beta)
end
function rand!(rng::AbstractRNG, v::AbstractVector, lmm::LMM)
    if !lmm.result.fit error("Model not fitted!") end
    rand!(rng, v, lmm, lmm.result.theta, lmm.result.beta)
end
rand!(v::AbstractVector, lmm::LMM) = rand!(default_rng(), v, lmm, lmm.result.theta, lmm.result.beta)
"""
    rand(rng::AbstractRNG, lmm::LMM{T}; theta) where T

Generate random responce vector 'lmm' model, theta covariance vector, and zero means.
"""
function rand(rng::AbstractRNG, lmm::LMM{T}, theta::AbstractVector) where T
    v = Vector{T}(undef, nobs(lmm))
    rand!(rng, v, lmm, theta)
end
function rand!(rng::AbstractRNG, v::AbstractVector, lmm::LMM{T}, theta::AbstractVector) where T
    n = length(lmm.covstr.vcovblock)
    v = Vector{T}(undef, nobs(lmm))
    tv = Vector{T}(undef, lmm.maxvcbl)
    gvec = gmatvec(theta, lmm.covstr)
    for i = 1:n
        q    = length(lmm.covstr.vcovblock[i])
        V    = zeros(q, q)
        vmatrix!(V, gvec, theta, lmm, i)
        if length(tv) != q resize!(tv, q) end
        Distributions.rand!(rng, MvNormal(Symmetric(V)), tv)
        v[lmm.covstr.vcovblock[i]] .= tv
    end
    v
end
rand(lmm::LMM, theta::AbstractVector) = rand(default_rng(), lmm, theta)
rand!(v::AbstractVector, lmm::LMM, theta::AbstractVector) = rand!(default_rng(), v, lmm, theta)
"""
    rand(rng::AbstractRNG, lmm::LMM{T}; theta, beta) where T

Generate random responce vector 'lmm' model, theta covariance vector and mean's vector.
"""
function rand(rng::AbstractRNG, lmm::LMM{T}, theta::AbstractVector, beta::AbstractVector) where T
    v = Vector{T}(undef, nobs(lmm))
    rand!(rng, v, lmm, theta, beta)
end
function rand!(rng::AbstractRNG, v::AbstractVector, lmm::LMM{T}, theta::AbstractVector, beta::AbstractVector) where T
    if length(beta) != size(lmm.data.xv, 2) error("Wrong beta length!") end
    n = length(lmm.covstr.vcovblock)
    tv = Vector{T}(undef, lmm.maxvcbl)
    m  = Vector{T}(undef, lmm.maxvcbl)
    gvec = gmatvec(theta, lmm.covstr)
    for i = 1:n
        q    = length(lmm.covstr.vcovblock[i])
        if length(tv) != q
            resize!(tv, q)
            resize!(m, q)
        end
        mul!(m, lmm.dv.xv[i], beta)
        V    = zeros(q, q)
        vmatrix!(V, gvec, theta, lmm, i)
        Distributions.rand!(rng, MvNormal(m, Symmetric(V)), tv)
        v[lmm.covstr.vcovblock[i]] .= tv
    end
    v
end
rand(lmm::LMM, theta::AbstractVector, beta::AbstractVector) = rand(default_rng(), lmm, theta, beta)
rand!(v::AbstractVector, lmm::LMM, theta::AbstractVector, beta::AbstractVector) = rand!(default_rng(), v, lmm, theta, beta)
