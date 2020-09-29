# fit.jl


function fit!(lmm::LMM)

    #θ = sqrt.([0.2075570458620876, 0.13517107978180684, 1.0, 0.02064464030301768, 0.04229496217886127])

    #Make varlink function

    fv  = varlinkvec(lmm.covstr.ct)
    fvr = varlinkrvec(lmm.covstr.ct)

    #remlfunc!(a,b,c,d) = Metida.fgh!(a, b, c, d; remlβcalc = x -> remlβcalc(Metida.varlinkvecapply!(x, fv)), remlcalc = (x,y) -> Metida.reml_sweep(lmm, x, Metida.varlinkvecapply!(y, fv)))

    #Optim options
    optmethod  = Optim.Newton()
    optoptions = Optim.Options(g_tol = 1e-12,
        iterations = 200,
        store_trace = true,
        show_trace = false,
        allow_f_increases = true)

    θ = rand(lmm.covstr.tl)
    θ .= 0.5
    #Optim.optimize(Optim.only_fgh!(remlfunc!), θ, optmethod, optoptions)

    #remlβoptim = x -> reml_sweep_β(lmm, varlinkvecapply!(x, fv))[1]
    #remlβoptim = x -> reml_sweep_β(lmm, x -> varlinkvecapply!(x, fv), x)[1]

    #Twice differentiable object
    td = TwiceDifferentiable(x -> reml_sweep_β(lmm, varlinkvecapply!(x, fv))[1], θ; autodiff = :forward)

    lmm.result.optim  = Optim.optimize(td, θ, optmethod, optoptions)

    lmm.result.theta            = varlinkvecapply!(deepcopy(Optim.minimizer(lmm.result.optim)), fv)

    #
    lmm.result.reml, lmm.result.beta, iC = reml_sweep_β(lmm, lmm.result.theta)

    lmm.result.h            = ForwardDiff.hessian(x -> reml_sweep_β(lmm, x)[1], lmm.result.theta)

    lmm.result.c            = pinv(iC)
    lmm.result.se           = diag(lmm.result.c) 
    lmm
end
