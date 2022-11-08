################################################################################
# Multiple imputation blocks and distributions

#using Bootstrap
#import Bootstrap: BootstrapSample, original, straps

struct MRS{D}
    block::Vector{Tuple{Int, Vector{Int}}}
    dist::Vector{D}
end

# unify with MetidaBase
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
        # check NaN values
        if any(x-> isnanm(x), rcol) error("Some values is NaN!") end
        # replace missing to NaN
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
struct BootstrapResult{T} #<: BootstrapSample
    lmm
    cn::Vector{String}
    beta::Vector{T}
    se::Vector{T}
    theta::Vector{T}
    bv::Vector{Vector{T}}  # Coef vector
    vv::Vector{Vector{T}}  # SE vector
    tv::Vector{Vector{T}}  # theta (var-cov) vecor
    rml::Vector{Int}       # iterations with warn and errors
    log::Vector{LMMLogMsg}
end

struct MIBootResult{T1, T2}
    mir::MILMMResult{T1}
    br::Vector{BootstrapResult{T2}}
    function MIBootResult(mir::MILMMResult{T1}, br::Vector{BootstrapResult{T2}}) where T1 where T2
        new{T1, T2}(mir, br)
    end
end

#=
Bootstrap.original(br::BootstrapResult) = br.beta

Bootstrap.original(br::BootstrapResult, idx::Int) = original(br)[idx]

#Bootstrap.straps(br::BootstrapResult) = bs.t1

Bootstrap.straps(br::BootstrapResult, idx::Int) = getindex.(br.bv, idx)

Bootstrap.nvar(br::BootstrapResult) = length(br.beta)
=#
nvar(br::BootstrapResult) = length(br.beta)
straps(br::BootstrapResult, idx::Int)  = getindex(br.bv, idx)
sdstraps(br::BootstrapResult, idx::Int) = getindex(br.vv, idx)

"""
    bootstrap(lmm::LMM; double = true, n = 100, varn = n, verbose = true, init = lmm.result.theta, rng = default_rng())

Parametric bootstrap.

!!! warning
    Experimental: API not stable, results not validated

- double - use double approach (default - true);
- n - number of bootstrap samples for coefficient estimtion;
- varn - number of bottstrap samples for varianvce estimation;
- verbose - show progress bar;
- init - initial values for lmm;
- rng - random number generator.

Parametric bootstrap based on generating random responce vector from known distribution, that given from fitted LMM model.
For one-stage bootstrap variance parameters and coefficients simulated in one step. For double bootstrap (two-tage) variance parameters simulated first,
than used for simulating coefficients on stage two.

```julia
lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0m;
random = Metida.VarEffect(Metida.@covstr(formulation|subject), Metida.CSH),
)
Metida.fit!(lmm)
bt = Metida.bootstrap(lmm; n = 1000, varn = 1000, double = true, rng = MersenneTwister(1234))
confint(bt)
```

See also: [`confint`](@ref), [`Metida.miboot`](@ref)
"""
function bootstrap(lmm::LMM; double = true, n = 100, varn = n, verbose = true, init = lmm.result.theta, del = true, rng = default_rng())
    isfitted(lmm) || throw(ArgumentError("lmm not fitted!"))
    if double
        return dbootstrap_(lmm; n = n, varn = varn, verbose = verbose, maxiter = 5, init = init,  del = del, rng = rng)
    else
        return bootstrap_(lmm; n = n, verbose = verbose, maxiter = 5, init = init,  del = del, rng = rng)
    end
end
"""
    Make distribution vector for each lmm block
"""
function make_dist_vec!(dist, lmm::LMM)
    nb   = nblocks(lmm)
    #dist = Vector{FullNormal}(undef, nb)
    Base.Threads.@threads for i = 1:nb
        q    = length(lmm.covstr.vcovblock[i])
        m    = Vector{Float64}(undef, q)
        mul!(m, lmm.dv.xv[i], lmm.result.beta)
        V    = zeros(Float64, q, q)
        vmatrix!(V, lmm.result.theta, lmm, i)
        dist[i] = MvNormal(m, Symmetric(V))
    end
    dist
end
"""
    Try to fit temprorary made lmm object
"""
function fit_lmm!(lmmb, init, dist, rng)
    nb = nblocks(lmmb)
    for j = 1:nb
        # Generate data from 'dist' distribution vector
        rand!(rng, dist[j], lmmb.dv.yv[j])
    end
    fit!(lmmb; init = init, hes = false)
    lmmb
end
"""

Check bootstrap results
"""
function check_lmm!(rml, log, lmmb, tlmm, tv, vi, i, ll, ul)
    if isfitted(lmmb)
        for v in vi
            vvar =  tv[v]^2 / tlmm[v]
            if !(ll < vvar < ul)
                push!(rml, i)
                lmmlog!(log, 1, LMMLogMsg(:WARN, "Itaration $i is suspisious: variance ratio = $vvar ."))
                break
            end
        end
    else
        push!(rml, i)
        lmmlog!(log, 1, LMMLogMsg(:ERROR, "Itaration $i was not successful."))
    end
end
"""
    Simple bootstrap.
"""
function bootstrap_(lmm::LMM{T}; n, verbose, maxiter, init, rng, del) where T
    #nb   = nblocks(lmm) thetalength(lmm) coefn(lmm)
    bv   = Vector{Vector{Float64}}(undef, coefn(lmm))
    vv   = Vector{Vector{Float64}}(undef, coefn(lmm))
    for i = 1:coefn(lmm)
        bv[i] = Vector{Float64}(undef, n)
        vv[i] = Vector{Float64}(undef, n)
    end
    tv   = Vector{Vector{Float64}}(undef, thetalength(lmm))
    for i = 1:thetalength(lmm)
        tv[i] = Vector{Float64}(undef, n)
    end
    rml  = Vector{Int}(undef, 0)
    log  = Vector{LMMLogMsg}(undef, 0)

   
    mres = ModelResult(false, nothing, fill(NaN, thetalength(lmm)), NaN, fill(NaN, coefn(lmm)), nothing, fill(NaN, coefn(lmm), coefn(lmm)), fill(NaN, coefn(lmm)), nothing, false)
    lmmb = LMM(lmm.model, lmm.mf, lmm.mm, lmm.covstr, lmm.data, LMMDataViews(lmm.dv.xv, deepcopy(lmm.dv.yv)), lmm.nfixed, lmm.rankx, mres, lmm.maxvcbl, Vector{LMMLogMsg}(undef, 0))

    vi   = findall(x-> x == :var, lmm.covstr.ct)
    tlmm = theta_(lmm) .^ 2
    ll   = quantile(FDist(1, 1), 1/n)
    ul   = quantile(FDist(1, 1), 1 - 1/n)
    dist = Vector{FullNormal}(undef, nblocks(lmm))
    make_dist_vec!(dist, lmm)

    lmmlog!(log, 1, LMMLogMsg(:INFO, "Start bootstrap..."))

    p = Progress(n, dt = 0.5,
            desc="Bootstrapping LMMs...",
            barglyphs=BarGlyphs('|','█', ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|',),
            barlen=20)
    for i = 1:n
        fit_lmm!(lmmb, init, dist, rng)
        c = coef_(lmmb)
        s = stderror_(lmmb)
        t = theta_(lmmb)
        for j = 1:coefn(lmm)
            bv[j][i] = c[j]
            vv[j][i] = s[j]
        end
        for j = 1:thetalength(lmm)
            tv[j][i] = t[j]
        end
        check_lmm!(rml, log, lmmb, tlmm, t, vi, i, ll, ul)
        lmmb.result.fit = false
        if verbose next!(p) end
    end
    lmmlog!(log, 1, LMMLogMsg(:INFO, "End bootstrap..."))
    if del && length(rml) > 0
        for j = 1:coefn(lmm)
            deleteat!(bv[j], rml)
            deleteat!(vv[j], rml)
        end
        for j = 1:thetalength(lmm)
            deleteat!(tv[j], rml)
        end
        lmmlog!(log, 1, LMMLogMsg(:WARN, "Some results ($(length(rml))) was deleted."))
    end
    BootstrapResult(lmm, coefnames(lmm), coef(lmm), stderror(lmm), theta(lmm), bv, vv, tv, rml, log)
end
"""
    Double bootstrap.
"""
function dbootstrap_(lmm::LMM{T}; n, varn, verbose, maxiter, init, rng, del) where T
    nb   = nblocks(lmm)
    #bv   = Vector{Vector{Float64}}(undef, coefn(lmm))

    # Final result vectors 
    tvr   = Vector{Vector{Float64}}(undef, thetalength(lmm))
    bvr   = Vector{Vector{Float64}}(undef, coefn(lmm))
    vvr   = Vector{Vector{Float64}}(undef, coefn(lmm))
    #=
    for i = 1:coefn(lmm)
        #bv[i] = Vector{Float64}(undef, n)
        vv[i] = Vector{Float64}(undef, n)
    end
    =#

    # Vectors for result from step I
    tv   = Vector{Vector{Float64}}(undef, varn)
    bv   = Vector{Vector{Float64}}(undef, varn)
    vv   = Vector{Vector{Float64}}(undef, varn)

    # Vectors for result from step II
    #tv2   = Vector{Vector{Float64}}(undef, varn)
    bv2   = Vector{Vector{Float64}}(undef, n)
    #vv2   = Vector{Vector{Float64}}(undef, varn)

    rml  = Vector{Int}(undef, 0)
    log  = Vector{LMMLogMsg}(undef, 0)

    mres = ModelResult(false, nothing, fill(NaN, thetalength(lmm)), NaN, fill(NaN, coefn(lmm)), nothing, fill(NaN, coefn(lmm), coefn(lmm)), fill(NaN, coefn(lmm)), nothing, false)
    lmmb = LMM(lmm.model, lmm.mf, lmm.mm, lmm.covstr, lmm.data, LMMDataViews(lmm.dv.xv, deepcopy(lmm.dv.yv)), lmm.nfixed, lmm.rankx, mres, lmm.maxvcbl, Vector{LMMLogMsg}(undef, 0))

    vi   = findall(x-> x == :var, lmm.covstr.ct)
    tlmm = theta_(lmm) .^ 2
    ll   = quantile(FDist(1, 1), 1/varn)
    ul   = quantile(FDist(1, 1), 1 - 1/varn)
    dist = Vector{FullNormal}(undef, nblocks(lmm))
    make_dist_vec!(dist, lmm)

    lmmlog!(log, 1, LMMLogMsg(:INFO, "Start bootstrap, step I..."))
    p = Progress(varn, dt = 0.5,
            desc="Bootstrapping I  LMMs...",
            barglyphs=BarGlyphs('|','█', ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|',),
            barlen=20)
    # STEP 1
    for i = 1:varn
        fit_lmm!(lmmb, init, dist, rng)
     
        #Fill vector for step I
        tv[i]  = theta(lmmb)
        bv[i]  = coef(lmmb)
        vv[i]  = stderror(lmmb)

        check_lmm!(rml, log, lmmb, tlmm, tv[i], vi, i, ll, ul)
        lmmb.result.fit = false
        if verbose next!(p) end
    end

    if length(rml) > 0
        deleteat!(tv, rml)
        deleteat!(bv, rml)
        lmmlog!(log, 1, LMMLogMsg(:WARN, "Step I: Some variance results ($(length(rml))) was deleted."))
        resize!(rml, 0)
    end
    lmmlog!(log, 1, LMMLogMsg(:INFO, "Start step II..."))
    
    # STEP 2
    p = Progress(n, dt=0.5,
            desc="Bootstrapping II LMMs...",
            barglyphs=BarGlyphs('|','█', ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|',),
            barlen=20)
    m   = Vector{T}(undef, lmm.maxvcbl)                 # means vector
    Vt  = Matrix{T}(undef, lmm.maxvcbl, lmm.maxvcbl)
    V   = view(Vt, 1:length(m), 1:length(m))
    tvl = length(tv)
    ll   = quantile(FDist(1, 1), 1/n)
    ul   = quantile(FDist(1, 1), 1 - 1/n)

    for i = 1:n

        # Iteration if number of bootstram more than on step I
        r = rem(i, tvl)
        ind = r == 0 ? tvl : r
        # Use theta from step I
        theta = tv[ind]
        # Use beta from step I
        beta  = bv[ind]
            # make distributions and generate responce
        for j = 1:nb
            q    = length(lmm.covstr.vcovblock[j])
            if length(m) != q resize!(m, q) end
            mul!(m, lmm.dv.xv[j], beta)
            if size(V, 1) != q
                V    = view(Vt, 1:q, 1:q)
            end
            fill!(V, zero(T))
            vmatrix!(V, theta, lmm, j)
            rand!(rng, MvNormal(m, Symmetric(V)), lmmb.dv.yv[j])
        end
        # fit
        fit!(lmmb; init = init, hes = false)

        bv2[i] = coef(lmmb)
   
        check_lmm!(rml, log, lmmb, tlmm, theta_(lmmb), vi, i, ll, ul)
        lmmb.result.fit = false
        if verbose next!(p) end
    end
    lmmlog!(log, 1, LMMLogMsg(:INFO, "End bootstrap..."))
    if del && length(rml) > 0
        deleteat!(bv2, rml)
        lmmlog!(log, 1, LMMLogMsg(:WARN, "Step II: Some results ($(length(rml))) was deleted."))
    end

    for j = 1:thetalength(lmm)
        tvr[j] = getindex.(tv, j) # theta from step I
    end
    for j = 1:coefn(lmm)
        vvr[j] = getindex.(vv, j) # coef-var from step I
        bvr[j] = getindex.(bv2, j) # beta from step II
    end
    
    BootstrapResult(lmm, coefnames(lmm), coef(lmm), stderror(lmm), theta(lmm), bvr, vvr, tvr, rml, log)
end
"""
    milmm(mi::MILMM; n = 100, verbose = true, rng = default_rng())

Multiple imputation.

!!! warning
    Experimental: API not stable, results not validated

For each subject random vector of missing values generated from distribution:

```math
X_{imp} \\sim N(\\mu_{miss \\mid obs}, \\Sigma_{miss \\mid obs})
```

```math
\\mu_{miss \\mid obs} = \\mu_1+ \\Sigma_{12} \\Sigma_{22}^{-1} (x_{obs}- \\mu_2)
```

```math
\\Sigma_{miss \\mid obs} = \\Sigma_{11}- \\Sigma_{12} \\Sigma_{22}^{-1} \\Sigma_{21}
```

```math
x = \\begin{bmatrix}x_{miss}  \\\\ x_{obs}  \\end{bmatrix};
\\mu = \\begin{bmatrix}\\mu_1  \\\\ \\mu_2  \\end{bmatrix};
\\Sigma = \\begin{bmatrix} \\Sigma_{11} & \\Sigma_{12} \\\\ \\Sigma_{21} & \\Sigma_{22} \\end{bmatrix}
```

Example:

```julia
lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0m;
random = Metida.VarEffect(Metida.@covstr(formulation|subject), Metida.CSH),
)
Metida.fit!(lmm)
mi = Metida.MILMM(lmm, df0m)
bm = Metida.milmm(mi; n = 100, rng = MersenneTwister(1234))
```
"""
function milmm(mi::MILMM; n = 100, verbose = true, rng = default_rng())
    lmm = Vector{LMM}(undef, n)
    rb  = getindex.(mi.mrs.block, 1)
    max = maximum(x->length(getindex(x, 2)), mi.mrs.block)
    ty  = Vector{Float64}(undef, max)
    for i = 1:n
        data, dv = generate_mi(rng, mi.data, mi.dv, mi.covstr.vcovblock, mi.mrs, rb, ty)
        lmmi = LMM(mi.lmm.model, mi.mf, mi.mm, mi.covstr, data, dv, mi.lmm.nfixed, mi.lmm.rankx, deepcopy(mi.lmm.result), mi.maxvcbl, mi.log)
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
    milmm(lmm::LMM, data; n = 100, verbose = true, rng = default_rng())

Multiple imputation in one step. `data` for `lmm` and for `milmm` should be the same,
if different data used resulst can be unpredictable.
"""
function milmm(lmm::LMM, data; n = 100, verbose = true, rng = default_rng())
    milmm(MILMM(lmm, data); n = n, verbose = verbose, rng = rng)
end

function milmm(lmm::LMM; n = 100, verbose = true, rng = default_rng())
    error("Method not defined!")
end
"""
    miboot(mi::MILMM; n = 100, verbose = true, rng = default_rng())

Multiple imputation with parametric bootstrap step.

!!! warning
    Experimental: API not stable, results not validated

Example:

```julia
lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0m;
random = Metida.VarEffect(Metida.@covstr(formulation|subject), Metida.CSH),
)
Metida.fit!(lmm)
mi = Metida.MILMM(lmm, df0m)
bm = Metida.miboot(mi; n = 100, rng = MersenneTwister(1234))
```
"""
function miboot(mi::MILMM{T}; n = 100, double = true, bootn = 100, varn = bootn, verbose = true, rng = default_rng()) where T
    mres = milmm(mi; n = n, verbose = verbose, rng = rng)
    br = Vector{BootstrapResult{T}}(undef, n)

    p = Progress(n, dt=0.5,
            desc="Bootstrap MI LMMs...",
            barglyphs=BarGlyphs('|','█', ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|',),
            barlen=20)

    for i = 1:n
        br[i] = bootstrap(mres.lmm[i]; double = double, n = bootn, varn = varn, verbose = false, init = mres.lmm[i].result.theta, rng = rng)
        if verbose next!(p) end
    end

    MIBootResult(mres, br)
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
    dist = Vector{FullNormal}(undef, length(mb))
    #Base.Threads.@threads
    for i in 1:length(mb)
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
        for m = 1:length(nm) - 1
            for n = m + 1:l
                mx[m,n] = v[nm[m], nm[n]]
            end
        end
        for m = 1:length(nm)
            mx[m,m] = v[nm[m], nm[m]]
        end
    else
        mx[1,1] = v[vec[1], vec[1]]
    end
    Symmetric(mx), nm
end
# conditional vovariance matrix
function mvconddist(mx::AbstractMatrix, nm::AbstractVector, vec::AbstractVector, beta::AbstractVector, xv::AbstractMatrix, yv::AbstractVector)  #₁₂₃₄¹²³⁴⁵⁶⁷⁸⁹⁺⁻
    m  = Vector{Float64}(undef, length(yv))
    mul!(m, xv, beta)
    q  = length(vec)
    N  = length(nm)
    if q < N
        μ  = m[nm]
        y  = yv[nm]
        p  = q + 1
        Σ₁ = mx[1:q, 1:q]
        Σ₁₂= view(mx, 1:q, p:N)
        Σ₂₂= mx[p:N, p:N]
        Σ⁻¹= inv(Σ₂₂)
        #Σ  = Symmetric(Σ₁ - Σ₁₂ * Σ⁻¹ * Σ₁₂')
        Σ  = Symmetric(mulαβαtinc!(Σ₁, Σ₁₂, Σ⁻¹, -1))
        μ₁ = μ[1:q]
        μ₂ = view(μ, p:N)
        a  = view(y, p:N)
        #M  = μ₁ - Σ₁₂ * Σ⁻¹ * (a - μ₂)
        M  = mulαβαtinc!(μ₁, Σ₁₂, Σ⁻¹, a, μ₂, -1)
    #p,N
        return MvNormal(M, Σ)
    else
        return MvNormal(m[nm], mx)
    end
end
# generate data views MI; X without changes
function generate_mi(rng, data, dv::LMMDataViews{T}, vcovblock, mrs, rb, ty)  where T
    y  = Vector{Vector{T}}(undef, length(vcovblock))
    yv = deepcopy(data.yv)
    for i = 1:length(vcovblock)
        if !(i in rb)
            y[i] = dv.yv[i]
        end
    end
    for i = 1:length(mrs.block)
        yt = deepcopy(dv.yv[mrs.block[i][1]])
        if length(ty) != length(mrs.block[i][2]) resize!(ty, length(mrs.block[i][2])) end
        yt[mrs.block[i][2]] .= rand!(rng, mrs.dist[i], ty)
        y[mrs.block[i][1]]   = yt
        yv[vcovblock[mrs.block[i][1]]]    .= yt
    end
    LMMData(data.xv, yv), LMMDataViews(dv.xv, y)
end
################################################################################
"""
    StatsBase.confint(br::BootstrapResult, n::Int; level::Float64=0.95, method=:jn)

Confidence interval for bootstrap result.

*method:
- :bp - bootstrap percentile;
- :rbp - reverse bootstrap percentile;
- :norm - Normal distribution;
- :bcnorm - Bias corrected Normal distribution;
- :jn - bias corrected (jackknife resampling).
"""
function StatsBase.confint(br::BootstrapResult, n::Int; level::Float64=0.95, method=:bp, delrml = false)
    v = straps(br, n)
    if length(br.rml) > 0 && delrml
         deleteat!(v, br.rml)
    end
    if method == :bp
        confint_q(br, v, n, 1-level)
    elseif method == :rbp
        confint_rq(br, v, n, 1-level)
    elseif method == :norm
        confint_n(br, v, n, 1-level)
    elseif method == :bcnorm
        confint_bcn(br, v, n, 1-level)
    elseif method == :var
        confint_var(br, v, n, 1-level)
    elseif method == :var2
        confint_var2(br, v, n, 1-level)
    elseif method == :jn
        confint_jn(br, v, n, 1-level)
    else
        error("Method unknown!")
    end
end
function StatsBase.confint(br::BootstrapResult; level::Float64=0.95, method=:bp, delrml = false)
    l = nvar(br)
    v = Vector{Tuple}(undef, l)
    for i = 1:l
        v[i] = confint(br, i; level = level, method = method, delrml = false)
    end
    v
end
####
function confint_q(bt::BootstrapResult, v, i::Int, alpha)
    (quantile(v, alpha/2), quantile(v, 1-alpha/2))
end
function confint_rq(bt::BootstrapResult, v, i::Int, alpha)
    (2bt.beta[i]-quantile(v, 1-alpha/2), 2bt.beta[i]-quantile(v, alpha/2))
end
function confint_n(bt::BootstrapResult, v, i::Int, alpha)
    d = Normal(bt.beta[i], sqrt(var(v)))
    (quantile(d, alpha/2), quantile(d, 1-alpha/2))
end
function confint_bcn(bt::BootstrapResult, v, i::Int, alpha)
    d = Normal(2bt.beta[i] - mean(v), sqrt(var(v)))
    (quantile(d, alpha/2), quantile(d, 1-alpha/2))
end
function confint_var(bt::BootstrapResult, v, i::Int, alpha)
    sd  = sqrt(mean(x->x^2, bt.vv[5]))
    q   = quantile(TDist(dof_residual(bt.lmm)), 1-alpha/2)
    (bt.beta[i] - sd*q, bt.beta[i] + sd*q)
end
function confint_var2(bt::BootstrapResult, v, i::Int, alpha)
    bias = abs(bt.beta[i] - mean(v))
    sd  = sqrt(mean(x->x^2, bt.vv[5]))
    q   = quantile(TDist(dof_residual(bt.lmm)), 1-alpha/2)
    (bt.beta[i] - sd*q - bias, bt.beta[i] + sd*q + bias)
end
function jn(v)
    s  = sum(v)
    n  = length(v)
    @. (s - v) / (n-1)
end
function confint_jn(bt::BootstrapResult, v, i::Int, alpha)

    m0    = bt.beta[i]
    n     = length(v)
    j     = jn(v)
    theta = sum(j)/n  # - CHECK THIS theta = m0
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
    print(io, "    Number of sets: $(length(mr.lmm))           ")
    lmmn = length(mr.lmm)
    if lmmn > 1
        println(io, "")
        cl   = coefn(mr.milmm.lmm)
        β    = zeros(cl)
        σ²   = zeros(cl)
        βii  = zeros(cl)
        βtv  = zeros(cl)
    #stderror_(lmm::LMM)
    #coef_(lmm::LMM)
        for i = 1:lmmn
            for c = 1:cl
                β[c]  += coef_(mr.lmm[i])[c]
                σ²[c] += stderror_(mr.lmm[i])[c]^2
            end
        end
        for c = 1:cl
            β[c]  = β[c]/lmmn
            σ²[c] = σ²[c]/lmmn
        end
        for i = 1:lmmn
            for c = 1:cl
                βii[c] += (coef_(mr.lmm[i])[c] - β[c])^2
            end
        end
        for c = 1:cl
            βii[c] = βii[c]/(lmmn-1)
            βtv[c] = σ²[c] + (1 + 1/lmmn)*βii[c]
        end
        mt = metida_table(coefnames(mr.milmm.lmm),  β, σ², βii, βtv, sqrt.(βtv); names = (Symbol("Coef. name"), Symbol("β"), Symbol("σ²"), Symbol("Inter-σ²"), Symbol("Total σ²"), Symbol("Total Std. Error")))
        show(io, mt)
    end
end

function tvlength(br::BootstrapResult)
    length(first(br.tv))
end
function bvlength(br::BootstrapResult)
    length(first(br.bv))
end
function msgnum(br::BootstrapResult, type)
    msgnum(br.log, type)
end
function Base.show(io::IO, br::BootstrapResult)
    println(io, "    Bootstrap results:                          ")
    println(io, "------------------------------------------------")
    println(io, "    Number of replications: $(tvlength(br))/$(bvlength(br))")
    println(io, "    Errors: $(msgnum(br, :ERROR))               ")
    println(io, "    Warnings: $(msgnum(br, :WARN))              ")
    print(io, "    Excluded/suspicious: $(length(br.rml))      ")
    if bvlength(br) > 1

        β    = zeros(nvar(br))
        σ    = zeros(nvar(br))
        vβ   = zeros(nvar(br))

        #βii  = zeros(nvar(br))
        #βtv  = zeros(nvar(br))
        for i = 1:nvar(br)
            isr    = straps(br, i)
            β[i]   = mean(isr)
            vβ[i]  = sqrt(var(isr))
            isr    = sdstraps(br, i)
            σ[i]   = sqrt(sum(x-> x^2, isr)/bvlength(br))
        end

        println(io, "")
        mt = metida_table(br.cn, br.beta, br.se,  β, vβ, σ; names = (Symbol("Coef. name"), Symbol("β*"), Symbol("σ*"), Symbol("β"), Symbol("var(β)"), Symbol("σ")))
        show(io, mt)
    end
end

function Base.show(io::IO, mb::MIBootResult)
    println(io, "    Multiple Imputation & Bootstrap             ")
    println(io, "------------------------------------------------")
    println(io, mb.mir)
    println(io, "    Total number of replications: $(sum(tvlength.(mb.br)))/$(sum(bvlength.(mb.br)))")
    println(io, "    Total Errors: $(sum(msgnum.(mb.br, :ERROR)))   ")
    print(io,   "    Total Warnings: $(sum(msgnum.(mb.br, :WARN)))")
end
