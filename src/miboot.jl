################################################################################
# Multiple imputation blocks and distributions
struct MRS{D}
    block::Vector{Tuple{Int, Vector{Int}}}
    dist::Vector{D}
end
isnanm(x) = isnan(x)
isnanm(x::Missing) = false
"""
    MILMM(lmm::LMM, data)

Multiple imputation model.
"""
struct MILMM{T} <: MetidaModel
    lmm::LMM{T}
    mf::ModelFrame
    mm::ModelMatrix
    covstr::CovStructure
    data::LMMData{T}
    dv::LMMDataViews{T}
    maxvcbl::Int
    mrs::MRS
    log::Vector{LMMLogMsg}
    function MILMM(lmm::LMM{T}, data) where T
        if !Tables.istable(data) error("Data not a table!") end
        if !isfitted(lmm) error("LMM should be fitted!") end
        tv = termvars(lmm.model.rhs)
        union!(tv, termvars(lmm.covstr.random))
        union!(tv, termvars(lmm.covstr.repeated))
        datam, data_ = StatsModels.missing_omit(NamedTuple{tuple(tv...)}(Tables.columntable(data)))
        rv           = termvars(lmm.model.lhs)[1]
        rcol         = Tables.getcolumn(data, rv)[data_]
        if any(x-> isnanm(x), rcol) error("Some values is NaN!") end
        replace!(rcol, missing => NaN)
        data         = merge(NamedTuple{(rv,)}((convert(Vector{Float64}, rcol),)), datam)
        lmmlog       = Vector{LMMLogMsg}(undef, 0)
        mf           = ModelFrame(lmm.mf.f, lmm.mf.schema, data, MetidaModel)
        mm           = ModelMatrix(mf)
        #mmf     = convert(Matrix{Float64}, mm.m)
        mmf          = mm.m
        #mmf    = float.(mm.m)
        lmmdata      = LMMData(mmf, data[rv])
        covstr       = CovStructure(lmm.covstr.random, lmm.covstr.repeated, data)
        dv           = LMMDataViews(mmf, lmmdata.yv, covstr.vcovblock)
        mb           = missblocks(dv.yv)
        dist         = mrsdist(lmm, mb, covstr, dv.xv, dv.yv)
        new{T}(lmm, mf, mm, covstr, lmmdata, dv, findmax(length, covstr.vcovblock)[1], MRS(mb, dist), lmmlog)
    end
end
struct MILMMResult{T}
    milmm::MILMM{T}
    lmm::Vector{LMM}
    function MILMMResult(milmm::MILMM{T}, lmm::Vector{LMM}) where T
        new{T}(milmm, lmm)
    end
end
struct BootstrapResult{T}
    beta::Vector{T}
    theta::Vector{T}
    bv::Vector{Vector{T}}
    tv::Vector{Vector{T}}
    rml::Vector{Int}
    log::Vector{String}
end
"""
    bootstrap(lmm::LMM{T}; double = false, n = 100, varn = n, verbose = true, rng = default_rng()) where T

Parametric bootstrap.
"""
function bootstrap(lmm::LMM{T}; double = false, n = 100, varn = n, verbose = true, rng = default_rng()) where T
    if double
        return dbootstrap_(lmm; n = n, varn = varn, verbose = verbose, rng = rng)
    else
        return bootstrap_(lmm; n = n, verbose = verbose, rng = rng)
    end
end

function bootstrap_(lmm::LMM{T}; n = 100, verbose = true, rng = default_rng()) where T
    nb   = nblocks(lmm)
    bv   = Vector{Vector{T}}(undef, n)
    tv   = Vector{Vector{T}}(undef, n)
    dist = Vector{FullNormal}(undef, nb)
    local rml  = Vector{Int}(undef, 0)
    local log  = Vector{String}(undef, 0)
    Base.Threads.@threads for i = 1:nb
        q    = length(lmm.covstr.vcovblock[i])
        m    = Vector{T}(undef, q)
        mul!(m, lmm.dv.xv[i], lmm.result.beta)
        V    = zeros(T, q, q)
        vmatrix!(V, lmm.result.theta, lmm, i)
        dist[i] = MvNormal(m, Symmetric(V))
    end
    lmmb = deepcopy(lmm)
    p = Progress(n, dt=0.5,
            desc="Bootstrapping LMMs...",
            barglyphs=BarGlyphs('|','█', ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|',),
            barlen=20)
    for i = 1:n
        local success = false
        local iter    = 5
        while !success && iter > 0
            Base.Threads.@threads  for j = 1:nb
                rand!(rng, dist[j], lmmb.dv.yv[j])
            end
            try
                fit!(lmmb; refitinit = true, hes = false)
                success = isfitted(lmmb)
            catch
                push!(log, "Error in iteration $i...")
            end
            iter -=1
        end
        if !isfitted(lmmb)
            push!(rml, i)
            push!(log, "Itaration $i was not successful...")
        end
        bv[i] = coef(lmmb)
        tv[i] = theta(lmmb)
        if verbose next!(p) end
    end
    BootstrapResult(coef(lmm), theta(lmm), bv, tv, rml, log)
end

function dbootstrap_(lmm::LMM{T}; n = 100, varn = 100, verbose = true, rng = default_rng()) where T
    nb   = nblocks(lmm)
    bv   = Vector{Vector{T}}(undef, n)
    tv   = Vector{Vector{T}}(undef, n)
    dist = Vector{MvNormal}(undef, nb)
    local rml  = Vector{Int}(undef, 0)
    local log  = Vector{String}(undef, 0)

    for i = 1:nb
        q    = length(lmm.covstr.vcovblock[i])
        m    = lmm.dv.xv[i] * lmm.result.beta
        V    = zeros(q, q)
        vmatrix!(V, lmm.result.theta, lmm, i)
        dist[i] = MvNormal(m, Symmetric(V))
    end
    lmmb = deepcopy(lmm)
    p = Progress(n, dt=0.5,
            desc="Bootstrapping I  LMMs...",
            barglyphs=BarGlyphs('|','█', ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|',),
            barlen=20)
    # STEP 1
    for i = 1:n
        local success = false
        local iter    = 5
        while !success && iter > 0
            Base.Threads.@threads for j = 1:nb
                rand!(rng, dist[j], lmmb.dv.yv[j])
            end
            try
                fit!(lmmb; refitinit = true, hes = false)
                success = isfitted(lmmb)
            catch
                push!(log, "Step I: Error in iteration $i...")
            end
            iter -=1
        end
        if !isfitted(lmmb)
            push!(rml, i)
            push!(log, "Step I: Itaration $i was not successful...")
        end
        tv[i] = theta(lmmb)
        if verbose next!(p) end
    end
    if length(rml) > 0
        deleteat!(tv, rml)
        push!(log, "Step I: Some variance results was deleted...")
    end
    # STEP 2
    p = Progress(n, dt=0.5,
            desc="Bootstrapping II LMMs...",
            barglyphs=BarGlyphs('|','█', ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|',),
            barlen=20)
    m   = Vector{T}(undef, lmm.maxvcbl)
    Vt  = Matrix{T}(undef, lmm.maxvcbl, lmm.maxvcbl)
    for i = 1:n
        local success = false
        local iter    = 5
        while !success && iter > 0
            theta = tv[rand(rng, 1:length(tv))]
            for j = 1:nb
                q    = length(lmm.covstr.vcovblock[j])
                if length(m) != q resize!(m, q) end
                mul!(m, lmm.dv.xv[j], lmm.result.beta)
                V    = view(Vt, 1:q, 1:q)
                fill!(V, zero(T))
                vmatrix!(V, theta, lmm, j)
                rand!(rng, MvNormal(m, Symmetric(V)), lmmb.dv.yv[j])
            end
            try
                fit!(lmmb; refitinit = true, hes = false)
                success = isfitted(lmmb)
            catch
                push!(log, "Error in iteration $i...")
            end
            iter -=1
        end
        if !isfitted(lmmb)
            push!(rml, i)
            push!(log, "Itaration $i was not successful...")
        end
        bv[i] = coef(lmmb)
        if verbose next!(p) end
    end
    BootstrapResult(coef(lmm), theta(lmm), bv, tv, rml, log)
end
"""
    milmm(mi::MILMM; n = 100, verbose = true, rng = default_rng())

Multiple imputation.
"""
function milmm(mi::MILMM; n = 100, verbose = true, rng = default_rng())
    lmm = Vector{LMM}(undef, n)
    rb  = getindex.(mi.mrs.block, 1)
    max = maximum(x->length(getindex(x, 2)), mi.mrs.block)
    ty  = Vector{Float64}(undef, max)
    for i = 1:n
        dv = generatedv(rng, mi.dv, mi.covstr.vcovblock, mi.mrs, rb, ty)
        lmmi = LMM(mi.lmm.model, mi.mf, mi.mm, mi.covstr, mi.data, dv, mi.lmm.nfixed, mi.lmm.rankx, deepcopy(mi.lmm.result), mi.maxvcbl, mi.log)
        lmm[i] = lmmi
    end
    p = Progress(n, dt = 0.5,
            desc="Computing MI LMMs...",
            barglyphs=BarGlyphs('|','█', ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|',),
            barlen=20)
    for i = 1:n
        fit!(lmm[i]; refitinit = true)
        if verbose next!(p) end
    end
    MILMMResult(mi, lmm)
end
"""
    miboot(mi::MILMM; n = 100, verbose = true, rng = default_rng())

Multiple imputation with parametric bootstrap step.
"""
function miboot(mi::MILMM; n = 100, varn = n, double = true, verbose = true, rng = default_rng())
    mres = milmm(mi; n = n)
    nb   = nblocks(mres.lmm[1])
    dist = Vector{MvNormal}(undef, nb)
    bv   = Vector{Vector}(undef, n)
    tv   = Vector{Vector}(undef, n)
    local rml  = Vector{Int}(undef, 0)
    local log  = Vector{String}(undef, 0)
    #V    = zeros(lmm[1].maxvcbl, lmm[1].maxvcbl)
    p = Progress(n, dt=0.5,
            desc="Bootstrap MI LMMs...",
            barglyphs=BarGlyphs('|','█', ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|',),
            barlen=20)
    for j = 1:n
        Base.Threads.@threads for i = 1:nb
            q    = length(mres.lmm[j].covstr.vcovblock[i])
            m    = mres.lmm[j].dv.xv[i] * mres.lmm[j].result.beta
            V    = zeros(q, q)
            vmatrix!(V, mres.lmm[j].result.theta, mres.lmm[j], i)
            dist[i] = MvNormal(m, Symmetric(V))
        end

        local success = false
        local iter    = 1
        while !success && iter < 10
            Base.Threads.@threads for i = 1:nb
                mres.lmm[j].dv.yv[i] = rand(rng, dist[i])
            end
            try
                fit!(mres.lmm[j]; refitinit = true, hes = false)
                success = isfitted(mres.lmm[j])
            catch
                push!(log, "Error in iteration $j...")
            end
            if !isfitted(mres.lmm[j])
                if !(j in rml) push!(rml, j) end
                push!(log, "Itaration $j (try $(iter)) was not successful...")
            end
            iter +=1
        end
        bv[j] = coef(mres.lmm[j])
        tv[j] = theta(mres.lmm[j])
        if verbose next!(p) end
    end
    BootstrapResult(coef(mi.lmm), theta(mi.lmm), bv, tv, rml, log)
end
# Finf all block with missing values
# NaN used for mising data
# return vector of Tuple (block number, missing values list)
function missblocks(yv)
    vec = Vector{Tuple{Int, Vector{Int}}}(undef, 0)
    for i in 1:length(yv)
        m = findall(x-> isnan(x), yv[i])
        if length(m) > 0
            push!(vec, (i,m))
        end
    end
    vec
end
# return distribution vector for
function mrsdist(lmm, mb, covstr, xv, yv)
    dist = Vector{MvNormal}(undef, length(mb))
    Base.Threads.@threads for i in 1:length(mb)
        v       = vmatrix(lmm.result.theta, covstr, mb[i][1])
        rv      = covmatreorder(v, mb[i][2])
        dist[i] = mvconddist(rv[1], rv[2], mb[i][2], lmm.result.beta, xv[mb[i][1]], yv[mb[i][1]])
    end
    dist
end
# reorder covariance matrix
function covmatreorder(v::AbstractMatrix{T}, vec) where T
    l  = size(v, 1)
    mx = zeros(T, l, l)
    nm = append!(deepcopy(vec), setdiff(collect(1:l), vec))
    if l > 1
        @inbounds for m = 1:length(nm) - 1
            @inbounds for n = m + 1:l
                mx[m,n] = v[nm[m], nm[n]]
            end
        end
        @inbounds for m = 1:length(nm)
            mx[m,m] = v[nm[m], nm[m]]
        end
    else
        @inbounds mx[1,1] = v[vec[1], vec[1]]
    end
    Symmetric(mx), nm
end
# conditional vovariance matrix
function mvconddist(mx, nm, vec, beta, xv::AbstractMatrix{T}, yv) where T #₁₂₃₄¹²³⁴⁵⁶⁷⁸⁹⁺⁻
    m  = Vector{T}(undef, length(yv))
    mul!(m, xv, beta)
    q  = length(vec)
    N  = length(nm)
    if q < N
        μ  = m[nm]
        y  = yv[nm]
        p  = q + 1
        Σ₁ = view(mx, 1:q, 1:q)
        Σ₁₂= view(mx, 1:q, p:N)
        Σ₂₂= view(mx, p:N, p:N)
        Σ⁻¹= inv(Matrix(Σ₂₂))
        Σ  = Σ₁ - Σ₁₂ * Σ⁻¹ * Σ₁₂'
        μ₁ = view(μ, 1:q)
        μ₂ = view(μ, p:N)
        a  = view(y, p:N)
        M  = μ₁ - Σ₁₂ * Σ⁻¹ * (a - μ₂)
    #p,N
        return MvNormal(M, Σ)
    else
        return MvNormal(m[nm], mx)
    end
end
# generate data views MI; X without changes
function generatedv(rng, dv::LMMDataViews{T}, vcovblock, mrs, rb, ty)  where T
    y = Vector{Vector{T}}(undef, length(vcovblock))
    #x = Vector{Matrix{nonmissingtype(M)}}(undef, length(vcovblock))
    for i = 1:length(vcovblock)
        if !(i in rb)
            #y[i] = convert(Vector{NMT}, dv.yv[i])
            y[i] = dv.yv[i]
            #x[i] = float.(dv.xv[i])
        end
    end
    @inbounds for i = 1:length(mrs.block)
        #yt = copy(dv.yv[mrs.block[i][1]])
        #yt = convert(Vector{NMT}, dv.yv[mrs.block[i][1]])
        yt = copy(dv.yv[mrs.block[i][1]])
        if length(ty) != length(mrs.block[i][2]) resize!(ty, length(mrs.block[i][2])) end
        yt[mrs.block[i][2]] .= rand!(rng, mrs.dist[i], ty)
        #rand!(mrs.dist[i], view(yt, mrs.block[i][2]))
        y[mrs.block[i][1]] = yt
        #x[mrs.block[i][1]] = float.(dv.xv[mrs.block[i][1]])
    end
    LMMDataViews(dv.xv, y)
end
################################################################################
"""
    StatsBase.confint(br::BootstrapResult, n::Int; level::Float64=0.95, method=:jn)

Confidence interval for bootstrap result.
"""
function StatsBase.confint(br::BootstrapResult, n::Int; level::Float64=0.95, method=:jn)
    if method == :bp
        confint_q(br, n, 1-level)
    elseif method == :rbp
        confint_rq(br, n, 1-level)
    elseif method == :norm
        confint_n(br, n, 1-level)
    elseif method == :jn
        confint_jn(br, n, 1-level)
    else
        error("Method unknown!")
    end
end
####
function confint_q(bt::BootstrapResult, i::Int, alpha)
    v = getindex.(bt.bv, i)
    if length(bt.rml) > 0
         deleteat!(v, bt.rml)
    end
    (quantile(v, alpha/2), quantile(v, 1-alpha/2))
end
function confint_rq(bt::BootstrapResult, i::Int, alpha)
    v = getindex.(bt.bv, i)
    if length(bt.rml) > 0
         deleteat!(v, bt.rml)
    end
    (2bt.beta[i]-quantile(v, 1-alpha/2), 2bt.beta[i]-quantile(v, alpha/2))
end
function confint_n(bt::BootstrapResult, i::Int, alpha)
    v = getindex.(bt.bv, i)
    if length(bt.rml) > 0
         deleteat!(v, bt.rml)
    end
    d = Normal(bt.beta[i], sqrt(var(v)))
    (quantile(d, alpha/2), quantile(d, 1-alpha/2))
end
function jn(v)
    s  = sum(v)
    n  = length(v)
    (s .- v) ./ (n-1)
end
function confint_jn(bt::BootstrapResult, i::Int, alpha)
    v     = getindex.(bt.bv, i)
    if length(bt.rml) > 0
         deleteat!(v, bt.rml)
    end
    m0    = bt.beta[i]
    n     = length(v)
    j     = jn(v)
    theta = sum(j)/n
    sum1 = sum(x->(theta-x)^3, j)
    sum2 = sum(x->(theta-x)^2, j)
    a    = sum1 / sqrt(sum2^3) / 6
    z0 = quantile(Normal(), count(x-> x < m0, v)/n)
    z1 = z0 + quantile(Normal(), alpha/2)
    z2 = z0 + quantile(Normal(), 1-alpha/2)
    a1 = cdf(Normal(), z0 + z1/(1-a*z1))
    a2 = cdf(Normal(), z0 + z2/(1-a*z2))
    a1, a2
    (quantile(v, a1), quantile(v, a2))
end
################################################################################
function Base.show(io::IO, milmm::MILMM)
    println(io, "    Linear Mixed Model - Multiple Imputation    ")
    println(io, "------------------------------------------------")
    println(io, "    Blocks with missing data: $(length(milmm.mrs.block))")
    m = sum(length.(getindex.(milmm.mrs.block, 2)))
    print(io, "    Missings: $(m) ($(round(m / nobs(milmm), digits = 2))%)")
end

function Base.show(io::IO, mr::MILMMResult)
    println(io, mr.milmm)
    println(io, "------------------------------------------------")
    println(io, "    Generated datasets:                         ")
    println(io, "    Number of sets: $(length(mr.lmm))           ")
end

function Base.show(io::IO, br::BootstrapResult)
    println(io, "    Bootstrap results:                          ")
    println(io, "------------------------------------------------")
    println(io, "    Number of replications: $(length(br.bv))    ")
    println(io, "    Errors: $(length(br.log))                   ")
    println(io, "    Excluded/suspicious: $(length(br.rml))      ")
end
