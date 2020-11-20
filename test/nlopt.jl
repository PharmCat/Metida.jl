using NLopt

lmm = Metida.LMM(@formula(var~sequence+period+formulation), df;
random = [Metida.VarEffect(Metida.@covstr(formulation), Metida.CSH)],
repeated = Metida.VarEffect(Metida.@covstr(formulation), Metida.VC),
subject = :subject)

lmmr = Metida.fit!(lmm)

################################################################################
opt = NLopt.Opt(:LN_BOBYQA, 5)
NLopt.ftol_rel!(opt, 1.0e-10)
NLopt.ftol_abs!(opt, 1.0e-10)
NLopt.xtol_rel!(opt, 1.0e-10)
NLopt.xtol_abs!(opt, 1.0e-10)
NLopt.initial_step!(opt, [0.005, 0.005, 0.005, 0.005, 0.005])
fv  = Metida.varlinkvec(lmm.covstr.ct)
init = deepcopy( lmm.result.optim.initial_x)
obj = (x,y) -> Metida.reml_sweep_β2(lmm, Metida.varlinkvecapply!(x, fv))[1]
NLopt.min_objective!(opt, obj)
#res = NLopt.optimize!(opt, init)
function fx(opt, i)
    init = deepcopy(i)
    NLopt.optimize!(opt, init)
end
@benchmark fx($opt, $init)
################################################################################
#:LN_BOBYQA
#:LN_NEWUOA
#:LN_NEWUOA_BOUND
#:LN_PRAXIS
#:LN_NELDERMEAD
#:LN_SBPLX
#:LN_COBYLA
################################################################################
