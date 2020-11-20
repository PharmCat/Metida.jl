using DataFrames, CSV, StatsModels, LinearAlgebra, ForwardDiff, BenchmarkTools, ForwardDiff, Optim
using NLopt
using SnoopCompile
using LineSearches
path    = dirname(@__FILE__)
cd(path)
df      = CSV.File(path*"/csv/df0.csv") |> DataFrame
categorical!(df, :subject);
categorical!(df, :period);
categorical!(df, :sequence);
categorical!(df, :formulation);

df.rvar = rand(20)

lmm = Metida.LMM(@formula(var~sequence+period+formulation), df;
random = [Metida.VarEffect(Metida.@covstr(formulation), Metida.CSH), Metida.VarEffect(Metida.@covstr(period+sequence), Metida.VC)],
)

lmm = Metida.LMM(@formula(var~sequence+period+formulation), df;
random = [Metida.VarEffect(Metida.@covstr(formulation), Metida.CSH), Metida.VarEffect(Metida.@covstr(period+sequence), Metida.VC)],
subject = :subject)

lmm = Metida.LMM(@formula(var~sequence+period+formulation), df;
random = [Metida.VarEffect(Metida.@covstr(formulation), Metida.CSH), Metida.VarEffect(Metida.@covstr(period), Metida.VC)],
)

lmm = Metida.LMM(@formula(var~sequence+period+formulation), df;
random = Metida.VarEffect(Metida.@covstr(formulation), Metida.VC),
subject = :subject)

lmm = Metida.LMM(@formula(var~sequence+period+formulation), df;
random = Metida.VarEffect(Metida.@covstr(formulation), Metida.SI),
repeated = Metida.VarEffect(Metida.@covstr(formulation), Metida.VC),
subject = :subject)

lmm = Metida.LMM(@formula(var~sequence+period+formulation), df;
random = Metida.VarEffect(Metida.@covstr(subject), Metida.SI),
repeated = Metida.VarEffect(Metida.@covstr(subject), Metida.SI))

lmm = Metida.LMM(@formula(var~sequence+period+formulation), df;
random = Metida.VarEffect(Metida.@covstr(subject), Metida.SI),
repeated = Metida.VarEffect(Metida.SI))

lmm = Metida.LMM(@formula(var~sequence+period+formulation), df;
random = [Metida.VarEffect(Metida.@covstr(period), Metida.SI), Metida.VarEffect(Metida.@covstr(formulation), Metida.SI)],
repeated = Metida.VarEffect(Metida.SI))

lmm = Metida.LMM(@formula(var~sequence+period+formulation), df;
random = Metida.VarEffect(Metida.@covstr(formulation), Metida.VC),
repeated = Metida.VarEffect(Metida.@covstr(formulation), Metida.VC),
subject = :subject)

lmm = Metida.LMM(@formula(var~sequence+period+formulation), df;
random = Metida.VarEffect(Metida.@covstr(formulation), Metida.CSH),
repeated = Metida.VarEffect(Metida.@covstr(formulation), Metida.VC),
subject = :subject)

lmm = Metida.LMM(@formula(var~sequence+period+formulation), df;
random = [Metida.VarEffect(Metida.@covstr(period), Metida.VC), Metida.VarEffect(Metida.@covstr(formulation), Metida.VC)],
)

lmm = Metida.LMM(@formula(var~sequence+period+formulation), df;
random = Metida.VarEffect(Metida.@covstr(formulation), Metida.VC),
subject = :subject)

lmm = Metida.LMM(@formula(var~sequence+period+formulation), df;
random = Metida.VarEffect(Metida.@covstr(formulation), Metida.VC))

lmm = Metida.LMM(@formula(var~sequence+period+formulation), df;
random = Metida.VarEffect(Metida.@covstr(formulation), Metida.VC),
repeated = Metida.VarEffect(Metida.@covstr(formulation), Metida.SI))

lmm = Metida.LMM(@formula(var~sequence+period+formulation), df;
random = Metida.VarEffect(Metida.@covstr(subject), Metida.ARH),
repeated = Metida.VarEffect(Metida.@covstr(subject), Metida.VC))


lmm = Metida.LMM(@formula(var~sequence+period+formulation), df;
random = [Metida.VarEffect(Metida.@covstr(formulation), Metida.CSH)],
repeated = Metida.VarEffect(Metida.@covstr(formulation), Metida.VC),
subject = :subject)

lmm = Metida.LMM(@formula(var~sequence+period+formulation), df;
random = [Metida.VarEffect(Metida.@covstr(formulation), Metida.CSH; subj = :subject),
          Metida.VarEffect(Metida.@covstr(period), Metida.VC; subj = :formulation)],
repeated = Metida.VarEffect(Metida.@covstr(formulation), Metida.SI),
)

lmm = Metida.LMM(@formula(var~sequence+period+formulation+rvar), df;
random = Metida.VarEffect(Metida.@covstr(rvar), Metida.VC),
repeated = Metida.VarEffect(Metida.@covstr(formulation), Metida.VC),
subject = :subject)


lmm = Metida.LMM(@formula(var~formulation&var2), df;
random = Metida.VarEffect(Metida.@covstr(var2), Metida.VC),
subject = :subject)



lmm = Metida.LMM(@formula(var~sequence+period+formulation), df;
random = [Metida.VarEffect(Metida.@covstr(formulation), Metida.CSH)],
repeated = Metida.VarEffect(Metida.@covstr(formulation), Metida.VC),
subject = [:sequence, :formulation])
Metida.fit!(lmm)

@benchmark Metida.fit!(lmm)

using MixedModels

fm = @formula(var ~ sequence+period+formulation + (0+formulation|subject))
mm = fit(MixedModel, fm, df, REML=true)


fm = @formula(var~sequence+period+formulation + (0+formulation|subject) + (1|period))
mm = fit(MixedModel, fm, df, REML=true)

fm = @formula(var ~ 1+rvar&formulation + (1|subject))
mm = fit(MixedModel, fm, df, REML=true)

θ = [0.02832, 0.02832, 0.02832, 0.02832]
θ = [0.02832, 0.02832, 0.02832]
Metida.gmat_base_z(θ, lmm.covstr)

lmm = Metida.LMM(@formula(var~sequence+period+formulation), df;
random = Metida.VarEffect(Metida.@covstr(formulation), Metida.CSH; subj = [:subject]))

lmm = Metida.LMM(@formula(var~sequence+period+formulation), df;
random = Metida.VarEffect(Metida.@covstr(formulation), Metida.CSH; subj = [:subject]),
repeated = Metida.VarEffect(Metida.@covstr(formulation), Metida.VC; subj =[:subject]))

lmm = Metida.LMM(@formula(var~sequence+period+formulation), df;
random = Metida.VarEffect(Metida.@covstr(formulation), Metida.CSH; subj = :subject),
repeated = Metida.VarEffect(Metida.@covstr(formulation), Metida.VC; subj = [:subject, :formulation]))

lmm = Metida.LMM(@formula(var~sequence+period+formulation), df;
random = Metida.VarEffect(Metida.@covstr(formulation), Metida.CSH; subj =  [:subject, :formulation]),
repeated = Metida.VarEffect(Metida.@covstr(formulation), Metida.VC; subj = [:subject, :formulation]))

lmm = Metida.LMM(@formula(var~sequence+period+formulation), df;
random = Metida.VarEffect(Metida.@covstr(formulation), Metida.CSH),
repeated = Metida.VarEffect(Metida.@covstr(formulation), Metida.VC),
subject = :subject)

lmm = Metida.LMM(@formula(var~sequence+period+formulation), df;
random = Metida.VarEffect(Metida.@covstr(formulation), Metida.CSH; subj = :subject),
repeated = Metida.VarEffect(Metida.@covstr(formulation), Metida.VC; subj = [:subject]),
subject = :subject)

lmm = Metida.LMM(@formula(var~sequence+period+formulation), df;
random = Metida.VarEffect(Metida.@covstr(formulation), Metida.CSH; subj = [:subject, :formulation]),
repeated = Metida.VarEffect(Metida.@covstr(formulation), Metida.VC; subj = [:subject, :formulation]),
subject =  [:subject, :formulation])

lmm = Metida.LMM(@formula(var~sequence+period+formulation), df;
random = Metida.VarEffect(Metida.@covstr(formulation), Metida.CSH; subj = :subject),
repeated = Metida.VarEffect(Metida.@covstr(formulation), Metida.VC; subj = [:subject, :formulation]),
subject = :subject)

Metida.fit!(lmm)
θ = [0.02832, 0.02832, 0.02832]
Metida.reml_sweep_β_ub(lmm, lmm.result.theta)
Metida.reml_sweep_β(lmm, lmm.result.theta)
Metida.reml_sweep_β2(lmm, lmm.result.theta)

θ = deepcopy(lmm.result.theta)

Metida.reml_sweep_β_ub(lmm, θ)
Metida.reml_sweep_β(lmm, θ)
Metida.reml_sweep_β2(lmm, θ)

@benchmark Metida.reml_sweep_β_ub(lmm, lmm.result.theta)

G = [0.3697 	-0.03628;
-0.03628 	0.003561]
mx = zeros(20, 20)

Metida.gmat_base_z!(mx, lmm.result.theta, lmm.covstr)
Metida.rmat_basep_z!(mx, lmm.result.theta, lmm.data.zrv, lmm.covstr)


Z = [1 1.0;
1  2.0;
1  3.0;
1  4.0]

R = Diagonal([0.02832, 0.02832, 0.02832, 0.02832])
