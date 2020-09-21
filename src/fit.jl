# fit.jl


function fit!(lmm::LMM)

    β   = [1.6099999999999992,
 -0.17083333333333264,
  0.14416666666666683,
  0.08000000000000033,
  0.14416666666666722,
 -0.0791666666666665]
    θ = sqrt.([0.4, 0.5, 0.1 ^ 2, 0.2, 0.3, 2., 1., 1., 1., 1.])
    mlf = x -> reml_sweep(lmm, β, x)



    optmethod  = Optim.Newton()

    optoptions = Optim.Options(g_tol = 1e-12,
        iterations = 10,
        store_trace = true,
        show_trace = false)



    #Optim.optimize(Optim.only_fgh!(fgh!), [0., 0.], optmethod, optoptions)

    mlf(θ)

end
