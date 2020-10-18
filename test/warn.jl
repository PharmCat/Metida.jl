

lmm = Metida.LMM(@formula(var~sequence+period+formulation), df;
random = [Metida.VarEffect(Metida.@covstr(formulation), Metida.CSH)],
repeated = Metida.VarEffect(Metida.@covstr(formulation), Metida.VC),
subject = :subject)

lmmr = Metida.fit!(lmm)

@code_warntype Metida.reml_sweep_β(lmm, lmm.result.theta)

@code_typed Metida.reml_sweep_β(lmm, lmm.result.theta)

V = [1 2 3 4; 1 2 3 3; 9 8 2 1; 1 2 1 2]
@code_warntype Metida.rmat_basep!(V, lmm.result.theta[lmm.covstr.tr[end]], lmm.data.zrv[1], lmm.covstr)

@code_warntype Metida.gmat_base(lmm.result.theta, lmm.covstr)
