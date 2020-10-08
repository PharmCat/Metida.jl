# fit.jl


function fit!(lmm::LMM{T}) where T

    #Make varlink function
    fv  = varlinkvec(lmm.covstr.ct)
    fvr = varlinkrvec(lmm.covstr.ct)

    #Optim options
    optmethod  = Optim.Newton()
    optoptions = Optim.Options(g_tol = 1e-12,
        iterations = 200,
        store_trace = true,
        show_trace = false,
        allow_f_increases = true)
    ############################################################################
    #Initial variance
    initθ = initvar(lmm.mf.data[lmm.mf.f.lhs.sym], lmm.mm.m)[1]
    θ  = zeros(T, lmm.covstr.tl)
    θ .= initθ / (length(lmm.covstr.random) + 1)
    for i = 1:length(θ)
        if lmm.covstr.ct[i] == :rho θ[i] = 0.5 end
    end
    varlinkvecapply!(θ, fvr)
    ############################################################################

    #Twice differentiable object
    td = TwiceDifferentiable(x -> reml_sweep_β(lmm, varlinkvecapply!(x, fv))[1], θ; autodiff = :forward)
    #Optimization object
    lmm.result.optim  = Optim.optimize(td, θ, optmethod, optoptions)
    #Theta (θ) vector
    lmm.result.theta  = varlinkvecapply!(deepcopy(Optim.minimizer(lmm.result.optim)), fv)
    #Hessian
    lmm.result.h      = ForwardDiff.hessian(x -> reml_sweep_β(lmm, x)[1], lmm.result.theta)
    #SVD decomposition
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
    lmm.result.reml, lmm.result.beta, iC = reml_sweep_β(lmm, lmm.result.theta)
    #Variance-vovariance matrix of β
    lmm.result.c            = pinv(iC)
    #SE
    lmm.result.se           = sqrt.(diag(lmm.result.c))
    #Fit true
    lmm.result.fit          = true
    lmm
end
