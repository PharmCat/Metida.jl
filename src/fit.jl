# fit.jl

fit_nlopt!(lmm::MetidaModel; kwargs...)  = error("MetidaNLopt not found. \n - Run `using MetidaNLopt` before.")

"""
    fit!(lmm::LMM{T};
    solver::Symbol = :default,
    verbose = :auto,
    varlinkf = :exp,
    rholinkf = :sigm,
    aifirst = :default,
    g_tol::Float64 = 1e-12,
    x_tol::Float64 = 1e-12,
    f_tol::Float64 = 1e-12,
    hes::Bool   = true,
    init = nothing,
    io::IO = stdout,
    ) where T

Fit LMM model.

* `solver` - :default / :nlopt / :cuda
* `verbose` - :auto / 1 / 2 / 3
* `varlinkf` - not implemented
* `rholinkf` - :sigm / :atan / :sqsigm / :psigm
* `aifirst` - first iteration with AI-like method - :default / :ai / :score
* `g_tol` - absolute tolerance in the gradient
* `x_tol` - absolute tolerance of theta vector
* `f_tol` - absolute tolerance in changes of the REML
* `hes` - calculate REML Hessian
* `init` - initial theta values
* `io` - uotput IO
"""
function fit!(lmm::LMM{T};
    solver::Symbol = :default,
    verbose = :auto,
    varlinkf = :exp,
    rholinkf = :sigm,
    aifirst        = :default,
    g_tol::Float64 = 1e-8,
    x_tol::Float64 = 1e-8,
    f_tol::Float64 = 1e-8,
    hes::Bool   = true,
    init = nothing,
    io::IO = stdout,
    time_limit = 120,
    iterations = 300
    ) where T

    if lmm.result.fit lmmlog!(io, lmm, verbose, LMMLogMsg(:INFO, "Refit model...")) end
    lmm.result.fit = false

    if solver == :nlopt
        return fit_nlopt!(lmm; solver = :nlopt, verbose = verbose, varlinkf = varlinkf, rholinkf = rholinkf, aifirst = aifirst, g_tol = g_tol, x_tol = x_tol, f_tol = f_tol, hes = false, init = init, io = io)
    elseif solver == :cuda
        return fit_nlopt!(lmm; solver = :cuda,  verbose = verbose, varlinkf = varlinkf, rholinkf = rholinkf, aifirst = aifirst, g_tol = g_tol, x_tol = x_tol, f_tol = f_tol, hes = false, init = init, io = io)
    end

    if verbose == :auto
        verbose = 1
    end

    # Optimization function
    optfunc = reml_sweep_β
    # Make data views
    data = LMMDataViews(lmm)
    #Optim options
    #alphaguess = InitialHagerZhang(α0=1.0) #25s
    #linesearch =  LineSearches.MoreThuente()
    #LineSearches.InitialQuadratic(α0 = 1.0, αmin = 1e-12, αmax = 1.0, ρ = 0.25, snap2one = (0.75, Inf))
    #LineSearches.BackTracking(order=3)
    optmethod  = Optim.Newton(;alphaguess = LineSearches.InitialHagerZhang(), linesearch = LineSearches.HagerZhang())
    #optmethod  = IPNewton()
    optoptions = Optim.Options(g_tol = g_tol, x_tol = x_tol, f_abstol = f_tol,
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
        initθ = sqrt(initvar(lmm.data.yv, lmm.mm.m)[1])/(length(lmm.covstr.random)+1)
        θ                      .= initθ
        for i = 1:length(θ)
            if lmm.covstr.ct[i] == :rho
                θ[i] = 1e-4
            end
        end
        lmmlog!(io, lmm, verbose, LMMLogMsg(:INFO, "Initial θ: "*string(θ)))
    end
    # Initial step with modified Newton method
    chunk  = ForwardDiff.Chunk{1}()
    if isa(aifirst, Bool)
        if aifirst aifirst == :ai else aifirst == :default end
    end
    ############################################################################
    if aifirst == :ai || aifirst == :score
        optstep!(lmm, data, θ; method = aifirst, maxopt = 10)
        lmmlog!(io, lmm, verbose, LMMLogMsg(:INFO, "First step with AI-like method ($aifirst), θ: "*string(θ)))
    end
    varlinkrvecapply!(θ, lmm.covstr.ct; rholinkf = rholinkf)

    # Twice differentiable object
    vloptf(x) = optfunc(lmm, data, varlinkvecapply(x, lmm.covstr.ct; rholinkf = rholinkf))[1]
    gcfg   = ForwardDiff.GradientConfig(vloptf, θ, chunk)
    hcfg   = ForwardDiff.HessianConfig(vloptf, θ, chunk)
    gfunc!(g, x) = ForwardDiff.gradient!(g, vloptf, x, gcfg, Val{false}())
    hfunc!(h, x) = begin
        ForwardDiff.hessian!(h, vloptf, x, hcfg, Val{false}())
    end
    td = TwiceDifferentiable(vloptf, gfunc!, hfunc!, θ)
    # Optimization object
    try
        lmm.result.optim  = Optim.optimize(td, θ, optmethod, optoptions)
    catch e
        lmmlog!(lmm, LMMLogMsg(:ERROR, "Newton method failed, try LBFGS. Error: $e"))
        #optmethod  = Optim.Newton(;alphaguess = LineSearches.InitialStatic(), linesearch = LineSearches.HagerZhang())
        #optmethod  = Optim.LBFGS(;alphaguess = LineSearches.InitialStatic(), linesearch = LineSearches.MoreThuente())
        optmethod  = Optim.LBFGS(;alphaguess = LineSearches.InitialStatic(), linesearch = LineSearches.Static())
        lmm.result.optim  = Optim.optimize(td, θ, optmethod, optoptions)
    end
        # Theta (θ) vector
    lmm.result.theta  = varlinkvecapply!(deepcopy(Optim.minimizer(lmm.result.optim)), lmm.covstr.ct; rholinkf = rholinkf)
    lmmlog!(io, lmm, verbose, LMMLogMsg(:INFO, "Resulting θ: "*string(lmm.result.theta)*"; $(Optim.iterations(lmm.result.optim)) iterations."))

        # -2 LogREML, β, iC
    lmm.result.reml, lmm.result.beta, iC, θ₃, noerrors = optfunc(lmm, data, lmm.result.theta)
        # Fit true
    if !isnan(lmm.result.reml) && !isinf(lmm.result.reml) && noerrors
        # Variance-vovariance matrix of β
        lmm.result.c            = inv(Matrix(iC))
        # SE
        if  !any(x-> x < 0.0, diag(lmm.result.c))
            lmm.result.se           = sqrt.(diag(lmm.result.c)) #ERROR: DomainError with -1.9121111845919027e-54
            if any(x-> x < 1e-8, lmm.result.se) && minimum(lmm.result.se)/maximum(lmm.result.se) < 1e-8 lmmlog!(io, lmm, verbose, LMMLogMsg(:WARN, "Some of the SE parameters is suspicious.")) end
            lmmlog!(io, lmm, verbose, LMMLogMsg(:INFO, "Model fitted."))
            lmm.result.fit      = true
        end
    end
    # Check G
    if lmm.covstr.random[1].covtype.s != :ZERO
        for i = 1:length(lmm.covstr.random)
            dg = det(gmatrix(lmm, i))
            if dg < 1e-8 lmmlog!(io, lmm, verbose, LMMLogMsg(:WARN, "det(G) of random effect $(i) is less 1e-08.")) end
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
        qrd = qr(lmm.result.h, Val(true))
        for i = 1:length(lmm.result.theta)
            if abs(qrd.R[i,i]) < 1E-8
                if lmm.covstr.ct[qrd.jpvt[i]] == :var
                    lmmlog!(io, lmm, verbose, LMMLogMsg(:WARN, "Hessian parameter (variation) QR.R diagonal value ($(qrd.jpvt[i])) is less than 1e-10."))
                elseif lmm.covstr.ct[qrd.jpvt[i]] == :rho
                    lmmlog!(io, lmm, verbose, LMMLogMsg(:WARN, "Hessian parameter (rho) QR.R diagonal value ($(qrd.jpvt[i])) is less than 1e-10."))
                end
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
    chunk  = ForwardDiff.Chunk{1}()
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
