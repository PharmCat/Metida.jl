using DataFrames, CSV, StatsModels, LinearAlgebra, ForwardDiff, BenchmarkTools, ForwardDiff, Optim
path    = dirname(@__FILE__)
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
random = [Metida.VarEffect(Metida.@covstr(formulation), Metida.CSH)],
repeated = Metida.VarEffect(Metida.@covstr(formulation), Metida.VC),
subject = :subject)

Metida.fit!(lmm)

Metida.gmat_blockdiag([1,2,3,4,5,6,7,8,9], lmm.covstr)

lmm.covstr.random[1].model

#Xv, Zv, yv = Metida.subjblocks(df, :subject, lmm.mm.m, lmm.Z, lmm.mf.data[lmm.model.lhs.sym])

qrd = qr(lmm.mm.m, Val(false))
β = inv(qrd.R)*qrd.Q'*lmm.mf.data[lmm.model.lhs.sym]
p = rank(lmm.mm.m)
θ = [0.2, 0.3, 0.4, 0.5, 0.1]
reml  = Metida.reml(yv, Zv, p, Xv, θ, β)
#-27.838887604599993
reml2 = Metida.reml_sweep2(lmm.data.yv, lmm.data.zv, lmm.rankx, lmm.data.xv, β, θ)

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


function gmat(θ::Vector{T}, zn) where T
    mx = Matrix{T}(undef, zn, zn)
    for m = 1:zn
        mx[m, m] = θ[m]
    end
    if zn > 1
        for m = 1:zn - 1
            for n = m + 1:zn
                mx[m, n] = mx[m, m] * mx[n, n] * θ[end]
            end
        end
    end
    for m = 1:zn
        mx[m, m] = mx[m, m]*mx[m, m]
    end
    Symmetric(mx)
end





function funcx(x)
    x[1]^2 + 3*x[2]^2 * x[3]^2
end

func2x = x -> funcx(x)

ForwardDiff.gradient(funcx, [1,2,3])
ForwardDiff.hessian(funcx, [1,2,3])

function fgh!(F,G,H,x)

  val = func2x(x)

  if G != nothing
      G .= ForwardDiff.gradient(funcx, x)
  end
  if H != nothing
      H .= ForwardDiff.hessian(funcx, x)
  end
  if F != nothing
    return val
  end
  nothing
end

optoptions = Optim.Options(g_tol = 1e-12,
    allow_f_increases = true)

opt = Optim.optimize(Optim.only_fgh!(fgh!), [3., 3., 3.], Optim.Newton(), optoptions)

opt = Optim.optimize( funcx, [3., 3., 3.], Optim.LBFGS(), optoptions)
