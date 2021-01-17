# fit.jl
#=
function gfunc!(g, x, f)
    chunk  = ForwardDiff.Chunk{1}()
    gcfg   = ForwardDiff.GradientConfig(f, x, chunk)
    ForwardDiff.gradient!(g, f, x, gcfg)
end
function hfunc!(h, x, f)
    chunk  = ForwardDiff.Chunk{1}()
    hcfg   = ForwardDiff.HessianConfig(f, x, chunk)
    ForwardDiff.hessian!(h, f, x, hcfg)
end
=#
fit_nlopt!(lmm::MetidaModel; kwargs...)  = error("MetidaNLopt not found. \n - Run `using MetidaNLopt` before.")

"""
    fit!(lmm::LMM{T}) where T

Fit LMM model.
"""
function fit!(lmm::LMM{T};
    solver::Symbol = :default,
    verbose = :auto,
    varlinkf = :exp,
    rholinkf = :sigm,
    aifirst::Bool = false,
    g_tol::Float64 = 1e-12,
    x_tol::Float64 = 1e-12,
    f_tol::Float64 = 1e-12,
    hcalck::Bool   = true,
    init = nothing) where T
    if solver == :nlopt
        return fit_nlopt!(lmm; solver = :nlopt, verbose = verbose, varlinkf = varlinkf, rholinkf = rholinkf, aifirst = aifirst, g_tol = g_tol, x_tol = x_tol, f_tol = f_tol, hcalck = false, init = init)
    elseif solver == :cuda
        return fit_nlopt!(lmm; solver = :cuda,  verbose = verbose, varlinkf = varlinkf, rholinkf = rholinkf, aifirst = aifirst, g_tol = g_tol, x_tol = x_tol, f_tol = f_tol, hcalck = false, init = init)
    end

    if verbose == :auto
        verbose = 1
    end
    #Make varlink function
    fv  = varlinkvec(lmm.covstr.ct)
    fvr = varlinkrvec(lmm.covstr.ct)
    # Optimization function
    if lmm.blocksolve
        optfunc = reml_sweep_β_b
        lmmlog!(lmm, verbose, LMMLogMsg(:INFO, "Solving by blocks..."))
    else
        optfunc = reml_sweep_β
    end

    #Optim options
    #alphaguess = InitialHagerZhang(α0=1.0) #25s
    #linesearch =  LineSearches.MoreThuente()
    #LineSearches.InitialQuadratic(α0 = 1.0, αmin = 1e-12, αmax = 1.0, ρ = 0.25, snap2one = (0.75, Inf))
    #LineSearches.BackTracking(order=3)
    optmethod  = Optim.Newton(;alphaguess = LineSearches.InitialHagerZhang(), linesearch = LineSearches.HagerZhang())
    #optmethod  = IPNewton()
    optoptions = Optim.Options(g_tol = g_tol, x_tol = x_tol, f_tol = f_tol,
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
    if isa(init, Vector{T}) && length(θ) == length(init)
        copyto!(θ, init)
    else
        initθ = sqrt(initvar(lmm.mf.data[lmm.mf.f.lhs.sym], lmm.mm.m)[1]/4)
        θ                      .= initθ
        for i = 1:length(θ)
            if lmm.covstr.ct[i] == :rho
                θ[i] = 0.0
                #lx[i] = -1.0
                #ux[i] = 1.0
            end
        end
        lmmlog!(lmm, verbose, LMMLogMsg(:INFO, "Initial θ: "*string(θ)))
    end
    #dfc = TwiceDifferentiableConstraints(lx, ux)
    #Initial step with modified Newton method
    ############################################################################
    if aifirst
        aif(x) = optfunc(lmm, x)[4]
        grf(x) = optfunc(lmm, x)[1]
        ai = ForwardDiff.hessian(aif, θ)
        gr = ForwardDiff.gradient(grf, θ)
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
                    θ[i] = 0.0
                end
            else
                if θ[i] < 0.01 θ[i] = initθ / 2 end
                if θ[i] > initθ*1.25 θ[i] = initθ*1.25 end
            end
        end
        lmmlog!(lmm, verbose, LMMLogMsg(:INFO, "First step with AI-like method, θ: "*string(θ)))
    end
    varlinkrvecapply2!(θ, lmm.covstr.ct)

    #Twice differentiable object

    vloptf(x) = optfunc(lmm, varlinkvecapply2(x, lmm.covstr.ct))[1]
    chunk  = ForwardDiff.Chunk{1}()
    gcfg   = ForwardDiff.GradientConfig(vloptf, θ, chunk)
    hcfg   = ForwardDiff.HessianConfig(vloptf, θ, chunk)
    gfunc!(g, x) = ForwardDiff.gradient!(g, vloptf, x, gcfg)
    hfunc!(h, x) = ForwardDiff.hessian!(h, vloptf, x, hcfg)
    td = TwiceDifferentiable(vloptf, gfunc!, hfunc!, θ)
    #td = TwiceDifferentiable(x ->optfunc(lmm, varlinkvecapply2(x, lmm.covstr.ct))[1], θ; autodiff = :forward)

    #Optimization object
    #td = TwiceDifferentiable(x ->optfunc(lmm, x)[1], θ; autodiff = :forward)
    #lmm.result.optim  = Optim.optimize(td, dfc, θ, optmethod, optoptions)
    try
        lmm.result.optim  = Optim.optimize(td, θ, optmethod, optoptions)
    catch
        optmethod  = Optim.Newton(;alphaguess = LineSearches.InitialStatic(), linesearch = LineSearches.HagerZhang())
        lmm.result.optim  = Optim.optimize(td, θ, optmethod, optoptions)
    end
    #Theta (θ) vector
    lmm.result.theta  = varlinkvecapply2!(deepcopy(Optim.minimizer(lmm.result.optim)), lmm.covstr.ct)
    try
        if hcalck
            #Hessian
            lmm.result.h      = ForwardDiff.hessian(x -> optfunc(lmm, x)[1], lmm.result.theta)
            #H positive definite check
            if !isposdef(lmm.result.h)
                lmmlog!(lmm, verbose, LMMLogMsg(:WARN, "Hessian is not positive definite."))
            end
            qrd = qr(lmm.result.h, Val(true))
            for i = 1:length(lmm.result.theta)
                if abs(qrd.R[i,i]) < 1E-10
                    if lmm.covstr.ct[qrd.jpvt[i]] == :var
                        lmmlog!(lmm, verbose, LMMLogMsg(:WARN, "Variation QR.R diagonal value ($(qrd.jpvt[i])) is less than 1e-10."))
                    elseif lmm.covstr.ct[qrd.jpvt[i]] == :rho
                        lmmlog!(lmm, verbose, LMMLogMsg(:WARN, "Rho QR.R diagonal value ($(qrd.jpvt[i])) is less than 1e-10."))
                    end
                end
            end
        end
        #-2 LogREML, β, iC
        lmm.result.reml, lmm.result.beta, iC, θ₃ = optfunc(lmm, lmm.result.theta)
        #Variance-vovariance matrix of β
        lmm.result.c            = pinv(iC)
        #SE
        lmm.result.se           = sqrt.(diag(lmm.result.c))
        #Fit true
        lmm.result.fit          = true
    catch
        #-2 LogREML, β, iC
        lmm.result.reml, lmm.result.beta, iC, θ₃ = optfunc(lmm, lmm.result.theta)
        #Fit false
        lmm.result.fit          = false
    end
    lmm
end
