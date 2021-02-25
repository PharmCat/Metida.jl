# fit.jl

fit_nlopt!(lmm::MetidaModel; kwargs...)  = error("MetidaNLopt not found. \n - Run `using MetidaNLopt` before.")

"""
    fit!(lmm::LMM{T};
    solver::Symbol = :default,
    verbose = :auto,
    varlinkf = :exp,
    rholinkf = :sigm,
    aifirst::Bool = false,
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
* `aifirst` - first iteration with AI-like method
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
    aifirst::Bool = false,
    g_tol::Float64 = 1e-8,
    x_tol::Float64 = 1e-8,
    f_tol::Float64 = 1e-8,
    hes::Bool   = true,
    init = nothing,
    io::IO = stdout
    ) where T

    if lmm.result.fit lmmlog!(io, lmm, verbose, LMMLogMsg(:INFO, "Refit model...")) end

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
        iterations = 300,
        time_limit = 120,
        store_trace = true,
        show_trace = false,
        allow_f_increases = true,
        extended_trace = true,
        callback = optim_callback)
    ############################################################################
    #Initial variance
    θ  = zeros(T, lmm.covstr.tl)
    #lx = similar(θ)
    #ux = similar(θ)
    #lx .= 0.0
    #ux .= Inf
    if isa(init, Vector{T})
        if length(θ) == length(init)
            copyto!(θ, init)
            lmmlog!(io, lmm, verbose, LMMLogMsg(:INFO, "Using provided θ: "*string(θ)))
        else
            error("init length $(length(init)) != θ length $(length(θ))")
        end
    else
        initθ = sqrt(initvar(lmm.data.yv, lmm.mm.m)[1]/4)
        θ                      .= initθ
        for i = 1:length(θ)
            if lmm.covstr.ct[i] == :rho
                θ[i] = 0.001
                #lx[i] = -1.0
                #ux[i] = 1.0
            end
        end
        lmmlog!(io, lmm, verbose, LMMLogMsg(:INFO, "Initial θ: "*string(θ)))
    end
    #dfc = TwiceDifferentiableConstraints(lx, ux)
    #Initial step with modified Newton method
    chunk  = ForwardDiff.Chunk{1}()
    ############################################################################
    if aifirst
        beta   = sweep_β(lmm, data, θ)
        aif(x) = sweep_ai(lmm, data, x, beta)
        grf(x) = optfunc(lmm, data, x)[1]
        aihcfg = ForwardDiff.HessianConfig(aif, θ, chunk)
        aigcfg = ForwardDiff.GradientConfig(grf, θ, chunk)
        ai = ForwardDiff.hessian(aif, θ, aihcfg, Val{false}())
        gr = ForwardDiff.gradient(grf, θ, aigcfg, Val{false}())
        initθ = copy(θ)
        try
            θ .-= (inv(ai) ./4 )*gr
        catch
            θ .-= (pinv(ai) ./4 )*gr
        end
        for i = 1:length(θ)
            if lmm.covstr.ct[i] == :rho
                if θ[i] > 0.99
                    θ[i] = 0.9
                elseif θ[i] < 0.0
                    θ[i] = 0.001
                end
            else
                if θ[i] < 0.01 θ[i] = initθ[i] / 2 end
                if θ[i] > initθ[i] * 1.25 θ[i] = initθ[i] * 1.25 end
            end
        end
        lmmlog!(io, lmm, verbose, LMMLogMsg(:INFO, "First step with AI-like method, θ: "*string(θ)))
    end
    varlinkrvecapply!(θ, lmm.covstr.ct; rholinkf = rholinkf)

    #Twice differentiable object

    vloptf(x) = optfunc(lmm, data, varlinkvecapply(x, lmm.covstr.ct; rholinkf = rholinkf))[1]
    gcfg   = ForwardDiff.GradientConfig(vloptf, θ, chunk)
    hcfg   = ForwardDiff.HessianConfig(vloptf, θ, chunk)
    gfunc!(g, x) = ForwardDiff.gradient!(g, vloptf, x, gcfg, Val{false}())
    hfunc!(h, x) = begin
        ForwardDiff.hessian!(h, vloptf, x, hcfg, Val{false}())
    end
    td = TwiceDifferentiable(vloptf, gfunc!, hfunc!, θ)
    #td = TwiceDifferentiable(x ->optfunc(lmm, varlinkvecapply2(x, lmm.covstr.ct))[1], θ; autodiff = :forward)

    #Optimization object
    #td = TwiceDifferentiable(x ->optfunc(lmm, x)[1], θ; autodiff = :forward)
    #lmm.result.optim  = Optim.optimize(td, dfc, θ, optmethod, optoptions)
    try
        lmm.result.optim  = Optim.optimize(td, θ, optmethod, optoptions)
    catch
        #optmethod  = Optim.Newton(;alphaguess = LineSearches.InitialStatic(), linesearch = LineSearches.HagerZhang())
        optmethod  = Optim.LBFGS(;alphaguess = LineSearches.InitialStatic(), linesearch = LineSearches.MoreThuente())
        lmm.result.optim  = Optim.optimize(td, θ, optmethod, optoptions)
    end
    #Theta (θ) vector
    lmm.result.theta  = varlinkvecapply!(deepcopy(Optim.minimizer(lmm.result.optim)), lmm.covstr.ct; rholinkf = rholinkf)
    lmmlog!(io, lmm, verbose, LMMLogMsg(:INFO, "Resulting θ: "*string(lmm.result.theta)))

        #-2 LogREML, β, iC
    lmm.result.reml, lmm.result.beta, iC, θ₃, noerrors = optfunc(lmm, data, lmm.result.theta)
        #Fit true
    if !isnan(lmm.result.reml) && !isinf(lmm.result.reml) && noerrors
        #Variance-vovariance matrix of β
        lmm.result.c            = pinv(iC)
        #SE
        lmm.result.se           = sqrt.(diag(lmm.result.c))
        lmmlog!(io, lmm, verbose, LMMLogMsg(:INFO, "Model fitted."))
        lmm.result.fit      = true
    else
        lmmlog!(io, lmm, verbose, LMMLogMsg(:INFO, "Model NOT fitted."))
        lmm.result.fit      = false
    end

    #Check G
    if lmm.covstr.random[1].covtype.s != :ZERO
        for i = 1:length(lmm.covstr.random)
            dg = det(gmatrix(lmm, i))
            if dg < 1e-08 lmmlog!(io, lmm, verbose, LMMLogMsg(:WARN, "det(G) of random effect $(i) is less 1e-08.")) end
        end
    end

    if hes && lmm.result.fit
            #Hessian
        lmm.result.h      = hessian(lmm, lmm.result.theta)
            #H positive definite check
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
    lmm
end
