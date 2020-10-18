

optmethod  = Optim.Newton()
optoptions = Optim.Options(g_tol = 1e-12,
    iterations = 300,
    store_trace = true,
    show_trace = true,
    allow_f_increases = true)
objopt = x -> Metida.reml_sweep_β(lmm, Metida.varlinkvecapply!(x, fv))[1]
O = Optim.optimize(objopt, deepcopy(lmm.result.optim.initial_x), Optim.Newton(); autodiff = :forward)
