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

Metida.fit!(lmm)

@benchmark Metida.fit!(lmm)
