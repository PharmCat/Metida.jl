# fit.jl

"""
    fit!(lmm::LMM{T}) where T

Fit LMM model.
"""
function fit!(lmm::LMM{T}; verbose::Symbol = :auto, varlinkf = :exp, rholinkf = :sigm) where T

    #Make varlink function
    fv  = varlinkvec(lmm.covstr.ct)
    fvr = varlinkrvec(lmm.covstr.ct)

    #Optim options
    #alphaguess = InitialHagerZhang(α0=1.0) #25s
    #linesearch =  LineSearches.MoreThuente()
    #LineSearches.InitialQuadratic(α0 = 1.0, αmin = 1e-12, αmax = 1.0, ρ = 0.25, snap2one = (0.75, Inf))
    #LineSearches.InitialQuadratic(α0 = 0.01, αmin = 1e-12, αmax = 0.5, ρ = 0.25, snap2one = (0.75, Inf))
    #LineSearches.InitialConstantChange()
    #LineSearches.BackTracking(order=3)
    optmethod  = Optim.Newton(;alphaguess = LineSearches.InitialHagerZhang(), linesearch = LineSearches.HagerZhang())
    #optmethod  = Optim.Newton(;alphaguess = LineSearches.InitialStatic(), linesearch = LineSearches.HagerZhang())
    optoptions = Optim.Options(g_tol = 1e-12, x_tol = 1e-10, f_tol = 1e-12,
        iterations = 300,
        store_trace = true,
        show_trace = false,
        allow_f_increases = true,
        callback = optim_callback)
    ############################################################################
    #Initial variance
    initθ = initvar(lmm.mf.data[lmm.mf.f.lhs.sym], lmm.mm.m)[1]
    θ  = zeros(T, lmm.covstr.tl)
    θ                      .= 0.01
    θ[lmm.covstr.tr[end]]  .= initθ
    #θ .= initθ / (length(lmm.covstr.random) + 1)
    for i = 1:length(θ)
        if lmm.covstr.ct[i] == :rho θ[i] = 0.0 end
    end
    #varlinkvecapply!(θ, fvr)
    varlinkrvecapply2!(θ, lmm.covstr.ct)
    ############################################################################
    if lmm.blocksolve optfunc = reml_sweep_β else optfunc = reml_sweep_β3 end
    #Twice differentiable object
    #td = TwiceDifferentiable(x ->optfunc(lmm, varlinkvecapply!(x, fv))[1], θ; autodiff = :forward)
    td = TwiceDifferentiable(x ->optfunc(lmm, varlinkvecapply2!(x, lmm.covstr.ct))[1], θ; autodiff = :forward)
    #Optimization object
    lmm.result.optim  = Optim.optimize(td, θ, optmethod, optoptions)
    #Theta (θ) vector
    #lmm.result.theta  = varlinkvecapply!(deepcopy(Optim.minimizer(lmm.result.optim)), fv)
    lmm.result.theta  = varlinkvecapply2!(deepcopy(Optim.minimizer(lmm.result.optim)), lmm.covstr.ct)
    #Hessian
    lmm.result.h      = ForwardDiff.hessian(x -> optfunc(lmm, x)[1], lmm.result.theta)
    #H positive definite check
    if !isposdef(lmm.result.h)
        push!(lmm.warn, "Hessian is not positive definite.")
    end
    #SVD decomposition
    try
        hsvd = svd(lmm.result.h)
        for i = 1:length(lmm.result.theta)
            if hsvd.S[i] < 1E-10
                hsvd.S[i] = 0
            end
        end
        rhsvd = hsvd.U * Diagonal(hsvd.S) * hsvd.Vt
        for i = 1:length(lmm.result.theta)
            if rhsvd[i,i] < 1E-10
                if lmm.covstr.ct[i] == :var
                    lmm.result.theta[i] = 0
                    push!(lmm.warn, "Variation parameter ($(i)) set to zero.")
                elseif lmm.covstr.ct[i] == :rho
                    push!(lmm.warn, "Rho SVD value ($(i)) is less than 1e-10.")
                end
            end
        end
        #-2 LogREML, β, iC
        lmm.result.reml, lmm.result.beta, iC = optfunc(lmm, lmm.result.theta)
        #Variance-vovariance matrix of β
        lmm.result.c            = pinv(iC)
        #SE
        lmm.result.se           = sqrt.(diag(lmm.result.c))
        #Fit true
        lmm.result.fit          = true
    catch
        #-2 LogREML, β, iC
        lmm.result.reml, lmm.result.beta, iC = optfunc(lmm, lmm.result.theta)
        #Fit false
        lmm.result.fit          = false
    end
    lmm
end

#=
function fit2!(lmm::LMM{T}) where T

    #Make varlink function
    fv  = varlinkvec(lmm.covstr.ct)
    fvr = varlinkrvec(lmm.covstr.ct)

    #Optim options
    optmethod  = Optim.Newton()
    optoptions = Optim.Options(g_tol = 1e-10,
        iterations = 300,
        store_trace = true,
        show_trace = false,
        allow_f_increases = true)
    ############################################################################
    #Initial variance
    initθ = initvar(lmm.mf.data[lmm.mf.f.lhs.sym], lmm.mm.m)[1]
    θ  = zeros(T, lmm.covstr.tl)
    θ                      .= 1.01
    θ[lmm.covstr.tr[end]]  .= initθ
    #θ .= initθ / (length(lmm.covstr.random) + 1)
    for i = 1:length(θ)
        if lmm.covstr.ct[i] == :rho θ[i] = 0.0 end
    end
    varlinkvecapply!(θ, fvr)
    ############################################################################

    opt = NLopt.Opt(:LN_BOBYQA,  thetalength(lmm))
    NLopt.ftol_rel!(opt, 1.0e-10)
    NLopt.ftol_abs!(opt, 1.0e-10)
    NLopt.xtol_rel!(opt, 1.0e-10)
    NLopt.xtol_abs!(opt, 1.0e-10)

    obj = (x,y) -> reml_sweep_β2(lmm, varlinkvecapply!(x, fv))[1]
    NLopt.min_objective!(opt, obj)
    result = NLopt.optimize!(opt, θ)
    #Optimization object
    #lmm.result.optim
    #Theta (θ) vector
    lmm.result.theta  = varlinkvecapply!(deepcopy(result[2]), fv)
    #Hessian
    lmm.result.h      = ForwardDiff.hessian(x -> reml_sweep_β2(lmm, x)[1], lmm.result.theta)
    #SVD decomposition
    try
        hsvd = svd(lmm.result.h)
        for i = 1:length(lmm.result.theta)
            if hsvd.S[i] < 1E-10 hsvd.S[i] = 0 end
        end
        rhsvd = hsvd.U * Diagonal(hsvd.S) * hsvd.Vt
        for i = 1:length(lmm.result.theta)
            if rhsvd[i,i] < 1E-10
                if lmm.covstr.ct[i] == :var lmm.result.theta[i] = 0 end
            end
        end
        #-2 LogREML, β, iC
        lmm.result.reml, lmm.result.beta, iC = reml_sweep_β2(lmm, lmm.result.theta)
        #Variance-vovariance matrix of β
        lmm.result.c            = pinv(iC)
        #SE
        lmm.result.se           = sqrt.(diag(lmm.result.c))
        #Fit true
        lmm.result.fit          = true
    catch
        #-2 LogREML, β, iC
        lmm.result.reml, lmm.result.beta, iC = reml_sweep_β2(lmm, lmm.result.theta)
        #Fit false
        lmm.result.fit          = false
    end
    lmm
end
=#
