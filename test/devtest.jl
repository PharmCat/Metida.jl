using DataFrames, CSV, StatsModels, LinearAlgebra, ForwardDiff, BenchmarkTools, ForwardDiff, Optim
#using NLopt
using SnoopCompile
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
#fieldnames(ContinuousTerm)

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

lmmr = Metida.fit!(lmm)
Optim.minimum(lmmr)

lmm = Metida.LMM(@formula(var~sequence+period+formulation), df;
random = [Metida.VarEffect(Metida.@covstr(period), Metida.VC), Metida.VarEffect(Metida.@covstr(formulation), Metida.VC)],
)

lmm = Metida.LMM(@formula(var~sequence+period+formulation), df;
random = Metida.VarEffect(Metida.@covstr(formulation), Metida.VC),
subject = :subject)

lmm = Metida.LMM(@formula(var~sequence+period+formulation), df;
random = Metida.VarEffect(Metida.@covstr(formulation), Metida.VC))


v   = copy(Optim.minimizer(lmmr))
Optim.minimum(lmmr)
rvf = varlinkvec(lmm.covstr.ct)
varlinkvecapply!(v, rvf)

be = ReplicateBE.rbe!(df0, dvar = :var, subject = :subject, formulation = :formulation, period = :period, sequence = :sequence, g_tol = 1e-10);



Metida.gmat_blockdiag([1,2,3,4,5,6,7,8,9], lmm.covstr)

lmm.covstr.random[1].model

Xv, Zv, rza, yv = Metida.subjblocks(df, :subject, lmm.mm.m, lmm.covstr.z, lmm.mf.data[lmm.mf.f.lhs.sym], lmm.covstr.rz)
qrd = qr(lmm.mm.m, Val(false))
β = inv(qrd.R)*qrd.Q'*lmm.mf.data[lmm.model.lhs.sym]
p = rank(lmm.mm.m)
θ = [0.2075570458620876, 0.13517107978180684, 1.0, 0.02064464030301768, 0.04229496217886127]
reml  = Metida.reml(yv, Zv, p, Xv, θ, β)
#-27.838887604599993
θ = sqrt.(θ)
reml2 = Metida.reml_sweep(lmm, β, θ)

β2 = [1.57749286231184, -0.1708333333333329, 0.19598442938454544, 0.14501427537631795, 0.1573631793251061, -0.07916666666666648]
reml2 = Metida.reml_sweep(lmm, β2, θ)

reml3 = Metida.reml_sweep_β(lmm, θ)


reml4 = Metida.reml_sweep(lmm, reml3[2], θ)

grad1 = ForwardDiff.gradient(x -> Metida.reml_sweep(lmm, reml3[2], x), θ)
grad2 = ForwardDiff.gradient(x -> Metida.reml_sweep_β(lmm, x)[1], θ)
hf = x -> Metida.reml_sweep(lmm, reml3[2], x)
hess1 = ForwardDiff.hessian(hf, θ)

G = Metida.gmat_blockdiag(θ, lmm.covstr)
G2 = Metida.gmat(θ[3:5])


grad = ForwardDiff.gradient(x -> Metida.reml(yv, Zv, p, Xv, x, β), θ)
#=
-18.039543789304744
-14.067773306660634
 -3.479116433698282
 -3.6012790991017023
  1.5443845439051467
=#
grad = ForwardDiff.gradient(x -> Metida.reml_sweep(lmm, β, x), θ)


hess = ForwardDiff.hessian(x -> Metida.reml_sweep(lmm, β, x), θ)


Metida.covmat_grad(Metida.vmat, Zv[1], θ)


Metida.reml_grad(yv, Zv, p, Xv, θ, β)
@btime Metida.reml_grad($yv, $Zv, $p, $Xv, $θ, $β)
@btime ForwardDiff.gradient($(x -> Metida.reml(yv, Zv, p, Xv, x, β)), $θ)


#=
y = [1,2,3,4,3]
X = [1 0 0 0;
     1 1 0 0;
     1 0 1 0;
     1 1 0 1;
     1 1 1 1]
β = [0.5, 0.7, 1.5, 0.1]
v = zeros(5)
y - X*β == Metida.mulr!(v, y, X, β)
=#
Z   = [1 0; 1 0; 0 1; 0 1]
G   = [1 0.5; 0.5 2]
R   = Diagonal([0.1, 04, 0.3, 0.9])
V   =  Z*G*Z'+R
r   = [1.0 , 2.0 , 3.0 , 4.0]
sV  = [V r; r' r'r]



optoptions = Optim.Options(g_tol = 1e-12,
    allow_f_increases = true)

opt = Optim.optimize(Optim.only_fgh!(fgh!), [3., 3., 3.], Optim.Newton(), optoptions)

opt = Optim.optimize( funcx, [3., 3., 3.], Optim.LBFGS(), optoptions)

Z = [1 0; 1 0; 0 1; 0 1]
G = [1 0.2; 0.2 0.9]
R = [1 0 0 0; 0 1 0 0; 0 0 .5 0; 0 0 0 .5]
X = [1 0 0; 1 1 1; 1 0 1; 1 0 1]
x = [1, 2, 3, 6]

Metida.mulαβαtc2(Z, G, R, X)


vl = [:var, :rho, :var]

vf = varlinkvec(vl)

ar = [1.,10.,3.]

varlinkvecapply!(ar, vf)


fv  = Metida.varlinkvec(lmm.covstr.ct)
fvr = Metida.varlinkrvec(lmm.covstr.ct)
θ = [0.2075570458620876, 0.13517107978180684, 0.99999, 0.02064464030301768, 0.04229496217886127]
θ = sqrt.(θ)
θ2 = Metida.varlinkvecapply!(θ, fvr)
#varlinkvecapply!(θ, fv)


b = Metida.reml_sweep_β(lmm, θ )[2]
grad2 = ForwardDiff.gradient(x -> remlβcalc(x)[1], θ)

grad2 = ForwardDiff.gradient(x -> Metida.reml_sweep_β(lmm, x)[1], θ)

grad2 = ForwardDiff.gradient(x -> remlβcalc(Metida.varlinkvecapply!(x, fv))[1], θ2)

grad2 = ForwardDiff.gradient(x -> remlcalc(b, x), θ)

GRAD = zeros(Float64, 5)
H    = zeros(Float64, 5, 5)
f    = remlfunc!(true, GRAD, H, θ2)

f    = remlfunc!(true, GRAD, H, Metida.varlinkvecapply!(a, fv))

optmethod  = Optim.Newton()

optoptions = Optim.Options(g_tol = 1e-12,
    iterations = 300,
    store_trace = true,
    show_trace = false,
    allow_f_increases = true)

θ2 = rand(5)
O1 = Optim.optimize(Optim.only_fgh!(remlfunc!), θ2, optmethod, optoptions)


remlβcalc2 = x -> Metida.reml_sweep_β(lmm, Metida.varlinkvecapply!(x, fv))[1]
#θ = rand(5)
td      = TwiceDifferentiable(remlβcalc2, θ2; autodiff = :forward)
O2 = Optim.optimize(td, θ2, optmethod, optoptions)

res = bboptimize(remlβcalc2; SearchRange = (-60., 60.0), NumDimensions = 5)

a = [0, 0, 0, 0]
A = [1 2 3 4; 1 2 3 3; 9 8 2 1; 1 2 1 2; 1 2 1 2]
B = [1, 2, 9, 1 , 2]
mulαtβinc!(a, A, B)

lmm = Metida.LMM(@formula(var~sequence+period+formulation), df;
random = Metida.VarEffect(Metida.@covstr(formulation), Metida.ARH),
repeated = Metida.VarEffect(Metida.@covstr(formulation), Metida.SI))

lmm = Metida.LMM(@formula(var~sequence+period+formulation), df;
random = [Metida.VarEffect(Metida.@covstr(formulation), Metida.SI), Metida.VarEffect(Metida.@covstr(formulation), Metida.VC)],
repeated = Metida.VarEffect(Metida.@covstr(formulation), Metida.VC),
subject = :subject)

lmmr = Metida.fit!(lmm)

@code_warntype mulαtβinc!(a, A, B)
@code_warntype remlβcalc2(θ2)
@code_warntype Metida.reml_sweep_β(lmm, lmm.result.theta)
@code_typed Metida.reml_sweep_β(lmm, lmm.result.theta)

@code_warntype Metida.gmat_blockdiag(lmm.result.theta, lmm.covstr)
@code_typed Metida.gmat_blockdiag(lmm.result.theta, lmm.covstr)

precompile(remlβcalc2, (Array{Float64,1}))

inf_timing = @snoopi Metida.fit!(lmm)
inf_timing = @snoopi tmin=0.01 include("test.jl")
pc = SnoopCompile.parcel(inf_timing)
SnoopCompile.write("precompile", pc)


SnoopCompile.@snoopc "$path/precompile/metida_compiles.log" begin
    using Metida, Pkg
    include(joinpath(dirname(dirname(pathof(Metida))), "test", "test.jl"))
end
data = SnoopCompile.read("$path/precompile/metida_compiles.log")
pc = SnoopCompile.parcel(reverse!(data[2]))
SnoopCompile.write("$path/precompile", pc)
