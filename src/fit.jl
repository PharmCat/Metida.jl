# fit.jl
#using DiffResults

fit_nlopt!(lmm::MetidaModel; kwargs...)  = error("MetidaNLopt not found. \n - Run `using MetidaNLopt` before.")

"""
    fit(::Type{T}, f::FormulaTerm, data;
    contrasts=Dict{Symbol,Any}(),  
    random::Union{Nothing, VarEffect, Vector{VarEffect}} = nothing, 
    repeated::Union{Nothing, VarEffect} = nothing,
    kwargs...)

Fit LMM model with @formula.

Keywords see [`fit!`](@ref)
"""
function fit(::Type{T}, f::FormulaTerm, data;
    contrasts=Dict{Symbol,Any}(),  
    random::Union{Nothing, VarEffect, Vector{VarEffect}} = nothing, 
    repeated::Union{Nothing, VarEffect} = nothing,
    kwargs...) where T <: LMM
    lmm = LMM(f, data, contrasts = contrasts, random = random, repeated = repeated)
    fit!(lmm; kwargs...)
end
"""
    fit(::Type{T}, f::LMMformula, data;
    contrasts=Dict{Symbol,Any}(),  
    kwargs...) where T <: LMM

Fit LMM model with [`@lmmformula`](@ref).

Keywords see [`fit!`](@ref)
"""
function fit(::Type{T}, f::LMMformula, data;
    contrasts=Dict{Symbol,Any}(),  
    kwargs...) where T <: LMM
    lmm = LMM(f, data, contrasts = contrasts)
    fit!(lmm; kwargs...)
end

"""
    fit!(lmm::LMM{T}; kwargs...
    ) where T

Fit LMM model.

# Keywords:

* `solver` - :default / :nlopt for using with MetidaNLopt.jl/ :cuda for using with MetidaCu.jl
* `verbose` - :auto / 1 / 2 / 3 - - 1 - only log,  2 - log and print,  3 - print only errors, other log, 0 (or any other value) - no logging
* `varlinkf` - :exp / :sq / :identity [ref](@ref varlink_header)
* `rholinkf` - :sigm / :atan / :sqsigm / :psigm
* `aifirst` - first iteration with AI-like method - :default / :ai / :score
* `aifmax` - maximum pre-optimization steps
* `g_tol` - absolute tolerance in the gradient
* `x_tol` - absolute tolerance of theta vector
* `f_tol` - absolute tolerance in changes of the REML
* `hes` - calculate REML Hessian
* `init` - initial theta values
* `io` - output IO
* `time_limit` - time limit = 120 sec
* `iterations` - maximum iterations = 300
* `refitinit` - true/false - if `true` - use last values for initial condition  (`false` by default)
* `optmethod` - Optimization method. Look at Optim.jl documentation. (Newton by default)
* `singtol` - singular tolerance = 1e-8
* `maxthreads` - maximum threads = min(num_cores(), Threads.nthreads())

"""
function fit!(lmm::LMM{T}; kwargs...) where T

    kwkeys = keys(kwargs)

    :solver ∈ kwkeys ? solver = kwargs[:solver] : solver = :default
    :verbose ∈ kwkeys ? verbose = kwargs[:verbose] : verbose = :auto
    :varlinkf ∈ kwkeys ? varlinkf = kwargs[:varlinkf] : varlinkf = :exp
    :rholinkf ∈ kwkeys ? rholinkf = kwargs[:rholinkf] : rholinkf = :sigm
    :aifirst ∈ kwkeys ? aifirst = kwargs[:aifirst] : aifirst = :default
    :aifmax ∈ kwkeys ? aifmax = kwargs[:aifmax] : aifmax = 10
    :g_tol ∈ kwkeys ? g_tol = kwargs[:g_tol] : g_tol = 1e-10
    :x_tol ∈ kwkeys ? x_tol = kwargs[:x_tol] : x_tol = 1e-10
    :f_tol ∈ kwkeys ? f_tol = kwargs[:f_tol] : f_tol = 1e-10
    :hes ∈ kwkeys ? hes = kwargs[:hes] : hes = true
    :init ∈ kwkeys ? init = kwargs[:init] : init = :nothing
    :io ∈ kwkeys ? io = kwargs[:io] : io = stdout
    :time_limit ∈ kwkeys ? time_limit = kwargs[:time_limit] : time_limit = 120
    :iterations ∈ kwkeys ? iterations = kwargs[:iterations] : iterations = 300
    :refitinit ∈ kwkeys ? refitinit = kwargs[:refitinit] : refitinit = false
    :optmethod ∈ kwkeys ? optmethod = kwargs[:optmethod] : optmethod = :default
    :singtol ∈ kwkeys ? singtol = kwargs[:singtol] : singtol = 1e-8
    :maxthreads ∈ kwkeys ? maxthreads = kwargs[:maxthreads] : maxthreads = min(num_cores(), Threads.nthreads())

    # If model was fitted, previous results can be used if `refitinit` == true
    # Before fitting clear log
    if lmm.result.fit
        if length(lmm.log) > 0 deleteat!(lmm.log, 1:length(lmm.log)) end #resize?
        lmmlog!(io, lmm, verbose, LMMLogMsg(:INFO, "Refit model..."))
        if refitinit
            lmmlog!(io, lmm, verbose, LMMLogMsg(:INFO, "Using previous initial parameters."))
            init = lmm.result.theta
        end
    end
    lmm.result.fit = false
    # Use default solver (Optim.jl with Newton)
    solver == :default || return fit_nlopt!(lmm; kwargs...)
    # Set verbose mode
    if verbose == :auto
        verbose = 1
    end
    # Optimization method
    if optmethod == :default
        optmethod  = NEWTON_OM
    end

    # Make data views
    #data = LMMDataViews(lmm)
    #Optim options
    optoptions = Optim.Options(g_abstol = g_tol, x_abstol = x_tol, f_abstol = f_tol,
        iterations = iterations,
        time_limit = time_limit,
        store_trace = true,
        show_trace = false,
        allow_f_increases = true,
        extended_trace = true,
        callback = optim_callback)
    ############################################################################
    # Initial variance
    θ  = zeros(T, lmm.covstr.tl)
    if isa(init, Vector{T})
        if length(θ) == length(init)
            copyto!(θ, init)
            lmmlog!(io, lmm, verbose, LMMLogMsg(:INFO, "Using provided θ: "*string(θ)))
        else
            error("init length $(length(init)) != θ length $(length(θ))")
        end
    else
        initθ = sqrt(initvar(lmm.data.yv, lmm.data.xv)[1])/(length(lmm.covstr.random)+1)
        for i = 1:length(θ)
            if lmm.covstr.ct[i] == :var
                θ[i] = initθ
            elseif lmm.covstr.ct[i] == :rho
                θ[i] = 1e-4
            elseif lmm.covstr.ct[i] == :theta
                θ[i] = 1.0
            end
        end
        lmmlog!(io, lmm, verbose, LMMLogMsg(:INFO, "Initial θ: "*string(θ)))
    end
    # Initial step with modified Newton method
    chunk  = ForwardDiff.Chunk{min(8, length(θ))}()
    if isa(aifirst, Bool)
        if aifirst aifirst == :ai else aifirst == :default end
    end
    ############################################################################
    if aifirst == :ai || aifirst == :score
        optstep!(lmm, lmm.dv, θ; method = aifirst, maxopt = aifmax)
        lmmlog!(io, lmm, verbose, LMMLogMsg(:INFO, "First step with AI-like method ($aifirst), θ: "*string(θ)))
    end
    varlinkrvecapply!(θ, lmm.covstr.ct; varlinkf = varlinkf, rholinkf = rholinkf)

    # Twice differentiable object / reml_sweep_β_nlopt
    vloptf(x)  = reml_sweep_β(lmm, lmm.dv, varlinkvecapply!(x, lmm.covstr.ct; varlinkf = varlinkf, rholinkf = rholinkf); maxthreads = maxthreads)[1]
    vloptfd(x) = reml_sweep_β_nlopt(lmm, lmm.dv, varlinkvecapply!(x, lmm.covstr.ct; varlinkf = varlinkf, rholinkf = rholinkf); maxthreads = maxthreads)[1]

    gcfg   = ForwardDiff.GradientConfig(vloptf, θ, chunk)
    hcfg   = ForwardDiff.HessianConfig(vloptf, θ, chunk)

    gfunc!(g, x) = begin
         ForwardDiff.gradient!(g, vloptf, x, gcfg)
    end
    fgfunc!(g, x) = begin
        gres = DiffResults.DiffResult(1.0, g)
        ForwardDiff.gradient!(gres, vloptf, x, gcfg)
        return DiffResults.value(gres)
    end
    hfunc!(h, x) = begin
        ForwardDiff.hessian!(h, vloptf, x, hcfg)
    end
    td = TwiceDifferentiable(vloptfd, gfunc!, fgfunc!, hfunc!, θ)
    # Optimization object
    try
        lmm.result.optim  = Optim.optimize(td, θ, optmethod, optoptions)
    catch e
        lmmlog!(lmm, LMMLogMsg(:ERROR, "Newton method failed, try BFGS. Error: $e"))
        optmethod  = BFGS_OM
        lmm.result.optim  = Optim.optimize(td, θ, optmethod, optoptions)
    end
        # Theta (θ) vector
    copyto!(lmm.result.theta, Optim.minimizer(lmm.result.optim))
    varlinkvecapply!(lmm.result.theta, lmm.covstr.ct; varlinkf = varlinkf, rholinkf = rholinkf)
    #lmm.result.theta  = varlinkvecapply!(deepcopy(Optim.minimizer(lmm.result.optim)), lmm.covstr.ct; varlinkf = varlinkf, rholinkf = rholinkf)
    lmmlog!(io, lmm, verbose, LMMLogMsg(:INFO, "Resulting θ: "*string(lmm.result.theta)*"; $(Optim.iterations(lmm.result.optim)) iterations."))

        # -2 LogREML, β, iC
    lmm.result.reml, beta, iC, θ₃, noerrors = reml_sweep_β(lmm, lmm.dv, lmm.result.theta)
    copyto!(lmm.result.beta, beta)
        # If errors in last evaluetion - log it.
    if !noerrors LMMLogMsg(:ERROR, "The last optimization step wasn't accurate. Results can be wrong!") end
        # Fit true
    if !isnan(lmm.result.reml) && !isinf(lmm.result.reml) && rank(iC) == size(iC, 1)
        # Variance-covariance matrix of β
        copyto!(lmm.result.c, inv(iC))
        # SE
        if  !any(x -> x < 0.0, diag(lmm.result.c))
            diag!(sqrt, lmm.result.se, lmm.result.c)
            if any(x-> x < singtol, lmm.result.se) && minimum(lmm.result.se)/maximum(lmm.result.se) < singtol lmmlog!(io, lmm, verbose, LMMLogMsg(:WARN, "Some of the SE parameters is suspicious.")) end
            if minimum(lmm.result.optim) == Inf 
                lmmlog!(io, lmm, verbose, LMMLogMsg(:INFO, "Minimum not estimated (Inf), model NOT fitted."))
                lmm.result.fit      = false
            elseif minimum(lmm.result.optim) === NaN
                lmmlog!(io, lmm, verbose, LMMLogMsg(:INFO, "Minimum not estimated (NaN), model NOT fitted."))
                lmm.result.fit      = false
            else
                lmmlog!(io, lmm, verbose, LMMLogMsg(:INFO, "Model fitted."))
                lmm.result.fit      = true
            end
        else
            lmmlog!(io, lmm, verbose,LMMLogMsg(:ERROR, "Some variance less zero: $(diag(lmm.result.c))."))
        end
    else
        lmmlog!(io, lmm, verbose, LMMLogMsg(:ERROR, "REML not estimated or final iteration completed with errors."))
    end
    # Check G
    lmm.result.ipd = true
    if !isa(lmm.covstr.random[1].covtype.s, ZERO)
        for i = 1:length(lmm.covstr.random)
            if isposdef(gmatrix(lmm, i)) == false
                lmm.result.ipd =  false
                lmmlog!(io, lmm, verbose, LMMLogMsg(:WARN, "Variance-covariance matrix (G) of random effect $(i) is not positive definite."))
            end
        end
    end

    # Check Hessian
    if hes && lmm.result.fit
            # Hessian
        lmm.result.h      = hessian(lmm, lmm.result.theta)
            # H positive definite check
        if !isposdef(Symmetric(lmm.result.h))
            lmmlog!(io, lmm, verbose, LMMLogMsg(:WARN, "Hessian is not positive definite."))
        end
        qrd = qr(lmm.result.h)
        for i = 1:length(lmm.result.theta)
            if abs(qrd.R[i,i]) < singtol
                lmmlog!(io, lmm, verbose, LMMLogMsg(:WARN, "Hessian parameter ($(lmm.covstr.ct[i])) QR.R diagonal value ($(i)) is less than $(singtol)."))
            end
        end
    end
    #
    if !lmm.result.fit
        lmmlog!(io, lmm, verbose, LMMLogMsg(:INFO, "Model NOT fitted."))
    end
    # return
    lmm
end

function optstep!(lmm, data, θ; method::Symbol = :ai, maxopt::Int=10)
    if method == :ai ai_func = sweep_ai else ai_func = sweep_score end
    reml, beta, θs₂, θ₃, rt = reml_sweep_β(lmm, data, θ)
    rt || error("Wrong initial conditions.")
    chunk  = ForwardDiff.Chunk{min(length(θ), 8)}()
    aif(x) = ai_func(lmm, data, x, beta)
    grf(x) = reml_sweep_β(lmm, data, x, beta)[1]
    aihcfg = ForwardDiff.HessianConfig(aif, θ, chunk)
    aigcfg = ForwardDiff.GradientConfig(grf, θ, chunk)
    ai = ForwardDiff.hessian(aif, θ, aihcfg, Val{false}())
    gr = ForwardDiff.gradient(grf, θ, aigcfg, Val{false}())
    θt = similar(θ)
    try
        mul!(θt, inv(ai), gr)
    catch
        mul!(θt, pinv(ai), gr)
    end
    for i = 1:length(θ)
        if lmm.covstr.ct[i] == :rho
            if θ[i] - θt[i] > 1.0 - eps() θt[i] = θ[i] - 1.0 + eps() elseif θ[i] - θt[i] < -1.0 + eps() θt[i] = θ[i] + 1.0 - eps() end
        else
            if θ[i] - θt[i] < eps() θt[i] = θ[i] / 2.0 end
        end
    end
    local maxopti = maxopt
    local remlc
    while maxopti > 0
        θr = θ - θt
        remlc, beta, θs₂, θ₃, rt = reml_sweep_β(lmm, data, θr)
        maxopti -= 1
        if rt && remlc < reml return copyto!(θ, θr), remlc, maxopt-maxopti, true else θt ./= 2.0 end
    end
    return θ, remlc, maxopt-maxopti, false
end
