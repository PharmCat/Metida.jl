using DataFrames, CSV, StatsModels, LinearAlgebra, ForwardDiff, BenchmarkTools
path    = dirname(@__FILE__)
df      = CSV.File(path*"/csv/df0.csv") |> DataFrame
categorical!(df, :subject);
categorical!(df, :period);
categorical!(df, :sequence);
categorical!(df, :formulation);

lmm = Metida.LMM(@formula(var~sequence+period+formulation), df;
random = Metida.VarEffect(Metida.@covstr(formulation), Metida.CSH)
)

Xv, Zv, yv = Metida.subjblocks(df, :subject, lmm.mm.m, lmm.Z, lmm.mf.data[lmm.model.lhs.sym])

qrd = qr(lmm.mm.m, Val(false))
β = inv(qrd.R)*qrd.Q'*lmm.mf.data[lmm.model.lhs.sym]
p = rank(lmm.mm.m)
θ = [0.2, 0.3, 0.4, 0.5, 0.1]
reml  = Metida.reml(yv, Zv, p, Xv, θ, β)
#-27.838887604599993
reml2 = Metida.reml_sweep(yv, Zv, p, Xv, θ, β)

grad = ForwardDiff.gradient(x -> Metida.reml(yv, Zv, p, Xv, x, β), θ)
#=
-18.039543789304744
-14.067773306660634
 -3.479116433698282
 -3.6012790991017023
  1.5443845439051467
=#
grad = ForwardDiff.gradient(x -> Metida.reml_sweep(yv, Zv, p, Xv, x, β), θ)


hess = ForwardDiff.hessian(x -> Metida.reml(yv, Zv, p, Xv, x, β), θ)


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
