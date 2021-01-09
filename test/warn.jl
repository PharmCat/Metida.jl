#Code warntype

@code_warntype Metida.reml_sweep_β3(lmm, lmm.result.theta)

@code_typed Metida.reml_sweep_β3(lmm, lmm.result.theta)

@code_warntype  Metida.gmat_base(lmm.result.theta, lmm.covstr)

@code_typed  Metida.gmat_base(lmm.result.theta, lmm.covstr)

V = [1. 2 3 4;
1 2 3 3;
9 8 2 1;
1 2 1 2]

@code_warntype Metida.rmat_basep!(V, lmm.result.theta[lmm.covstr.tr[end]], view(lmm.covstr.rz, lmm.data.block[1],:), lmm.covstr)

@code_warntype Metida.gmat_base_z2!(V, lmm.result.theta, lmm.covstr, lmm.data.block[1], lmm.covstr.sblock[1])

@code_warntype Metida.rmat_basep_z2!(V, lmm.result.theta[lmm.covstr.tr[end]], lmm.covstr, lmm.data.block[1], lmm.covstr.sblock[1])

@code_warntype Metida.rmatp_si!(V, lmm.result.theta[lmm.covstr.tr[end]], view(lmm.covstr.rz, lmm.covstr.block[end][1], :), lmm.covstr.repeated.covtype)

@code_typed Metida.rmatp_si!(V, lmm.result.theta[lmm.covstr.tr[end]], view(lmm.covstr.rz, lmm.covstr.block[end][1], :), lmm.covstr.repeated.covtype)

@code_warntype Metida.rmatp_diag!(V, lmm.result.theta[lmm.covstr.tr[end]], view(lmm.covstr.rz, lmm.covstr.block[end][1], :), lmm.covstr.repeated.covtype)

@code_warntype Metida.rmatp_csh!(V, lmm.result.theta[lmm.covstr.tr[end]], view(lmm.covstr.rz, lmm.covstr.block[end][1], :), lmm.covstr.repeated.covtype)

@code_warntype Metida.rmatp_cs!(V, lmm.result.theta[lmm.covstr.tr[end]], view(lmm.covstr.rz, lmm.covstr.block[end][1], :), lmm.covstr.repeated.covtype)

@code_warntype Metida.rmatp_ar!(V, lmm.result.theta[lmm.covstr.tr[end]], view(lmm.covstr.rz, lmm.covstr.block[end][1], :), lmm.covstr.repeated.covtype)

@code_warntype Metida.rmatp_arh!(V, lmm.result.theta[lmm.covstr.tr[end]], view(lmm.covstr.rz, lmm.covstr.block[end][1], :), lmm.covstr.repeated.covtype)


G = [1. 2.;
2. 3.]
@code_warntype Metida.gmat_si!(G, lmm.result.theta[lmm.covstr.tr[1]], lmm.covstr.q[1], lmm.covstr.random[1].covtype)

@code_warntype Metida.gmat_diag!(G, lmm.result.theta[lmm.covstr.tr[1]], lmm.covstr.q[1], lmm.covstr.random[1].covtype)

@code_warntype Metida.gmat_csh!(G, lmm.result.theta[lmm.covstr.tr[1]], lmm.covstr.q[1], lmm.covstr.random[1].covtype)

@code_warntype Metida.gmat_cs!(G, lmm.result.theta[lmm.covstr.tr[1]], lmm.covstr.q[1], lmm.covstr.random[1].covtype)

@code_warntype Metida.gmat_ar!(G, lmm.result.theta[lmm.covstr.tr[1]], lmm.covstr.q[1], lmm.covstr.random[1].covtype)
