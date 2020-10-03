using Metida, StatsBase, StatsModels, CSV, DataFrames
df = CSV.File(dirname(pathof(Metida))*"\\..\\test\\csv\\df0.csv") |> DataFrame

# EXAMPLE 1
################################################################################
################################################################################
################################################################################
#=
PROC MIXED data=df0;
CLASSES subject sequence period formulation;
MODEL  var = sequence period formulation/ DDFM=SATTERTH s;
RANDOM  formulation/TYPE=CSH SUB=subject G V;
REPEATED/GRP=formulation SUB=subject R;
RUN;

REML: 10.06523862
=#

lmm = LMM(@formula(var ~ sequence + period + formulation), df;
random   = VarEffect(@covstr(formulation), CSH),
repeated = VarEffect(@covstr(formulation), VC),
subject  = :subject)

fit!(lmm)
#=
Linear Mixed Model: var ~ sequence + period + formulation
Random 1:
   Model: formulation
   Type: HeterogeneousCompoundSymmetry (3)
   Coefnames: ["formulation: 1", "formulation: 2"]
Repeated:
   Model: formulation
   Type: VarianceComponents (2)
   Coefnames: ["formulation: 1", "formulation: 2"]

Status: converged

   -2 logREML: 10.0652

   Fixed effects:

Name             Estimate     SE
(Intercept)      1.57749      0.334543
sequence: 2      -0.170833    0.384381
period: 2        0.195984     0.117228
period: 3        0.145014     0.109171
period: 4        0.157363     0.117228
formulation: 2   -0.0791667   0.0903709

Random effects:

   θ vector: [0.455584, 0.367656, 1.0, 0.143682, 0.205657]

Random 1   formulation: 1   var   0.207557
Random 1   formulation: 2   var   0.135171
Random 1   Rho              rho   1.0
Repeated   formulation: 1   var   0.0206445
Repeated   formulation: 2   var   0.0422948
=#

# EXAMPLE 2
################################################################################
################################################################################
################################################################################
#=
PROC MIXED data=df0;
CLASSES subject sequence period formulation;
MODEL  var = sequence period formulation/ DDFM=SATTERTH s;
RANDOM  formulation/TYPE=VC SUB=subject G V;
REPEATED/GRP=formulation SUB=subject R;
RUN;

REML: 16.06148160
=#

lmm = LMM(
    @formula(var ~ sequence + period + formulation), df;
    random   = VarEffect(@covstr(formulation), SI),
    repeated = VarEffect(@covstr(formulation), VC),
    subject  = :subject,
)
fit!(lmm)
#=
Linear Mixed Model: var ~ sequence + period + formulation
Random 1:
   Model: formulation
   Type: ScaledIdentity (1)
   Coefnames: ["formulation: 1", "formulation: 2"]
Repeated:
   Model: formulation
   Type: VarianceComponents (2)
   Coefnames: ["formulation: 1", "formulation: 2"]

Status: converged

   -2 logREML: 16.0615

   Fixed effects:

Name             Estimate     SE
(Intercept)      1.57212      0.305807
sequence: 2      -0.170833    0.279555
period: 2        0.204087     0.289957
period: 3        0.155769     0.11308
period: 4        0.160015     0.289957
formulation: 2   -0.0791667   0.279555

Random effects:

   θ vector: [0.412436, 0.145184, 0.220819]

Random 1   Var              var   0.170103
Repeated   formulation: 1   var   0.0210784
Repeated   formulation: 2   var   0.048761
=#

#EXAMPLE 3
################################################################################
################################################################################
################################################################################
#=
PROC MIXED data=df0;
CLASSES subject sequence period formulation;
MODEL  var = sequence period formulation/ DDFM=SATTERTH s;
RANDOM  subject/TYPE=VC G V;
RUN;

REML: 10.86212458
=#

lmm = LMM(@formula(var ~ sequence + period + formulation), df;
    random = VarEffect(@covstr(subject), SI)
    )
fit!(lmm)
#=
Linear Mixed Model: var ~ sequence + period + formulation
Random 1:
   Model: subject
   Type: ScaledIdentity (1)
   Coefnames: ["subject: 1", "subject: 2", "subject: 3", "subject: 4", "subject: 5"]
Repeated:
   Model: nothing
   Type: ScaledIdentity (1)
   Coefnames: -

Status: converged

   -2 logREML: 10.8621

   Fixed effects:

Name             Estimate     SE
(Intercept)      1.61         0.309774
sequence: 2      -0.170833    0.383959
period: 2        0.144167     0.116706
period: 3        0.08         0.115509
period: 4        0.144167     0.116706
formulation: 2   -0.0791667   0.0833617

Random effects:

   θ vector: [0.410574, 0.182636]

Random 1   Var   var   0.168571
Repeated   Var   var   0.0333559
=#

#EXAMPLE 4
################################################################################
################################################################################
################################################################################
#=
PROC MIXED data=df0;
CLASSES subject sequence period formulation;
MODEL  var = sequence period formulation/ DDFM=SATTERTH s;
RANDOM  period/TYPE=VC G V;
RANDOM  formulation/TYPE=VC G V;
RUN;

REML: 25.12948063
=#
lmm = LMM(
    @formula(var ~ sequence + period + formulation), df;
    random   = [VarEffect(@covstr(period), VC), VarEffect(@covstr(formulation), VC)]
)
fit!(lmm)

#=
Linear Mixed Model: var ~ sequence + period + formulation
Random 1:
   Model: period
   Type: VarianceComponents (4)
   Coefnames: ["period: 1", "period: 2", "period: 3", "period: 4"]
Random 2:
   Model: formulation
   Type: VarianceComponents (2)
   Coefnames: ["formulation: 1", "formulation: 2"]
Repeated:
   Model: nothing
   Type: ScaledIdentity (1)
   Coefnames: -

Status: converged

   -2 logREML: 25.1295

   Fixed effects:

Name             Estimate     SE
(Intercept)      1.61         0.249491
sequence: 2      -0.170833    0.192487
period: 2        0.144167     0.269481
period: 3        0.08         0.266717
period: 4        0.144167     0.269481
formulation: 2   -0.0791667   0.192487

Random effects:

   θ vector: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.421717]

Random 1   period: 1        var   0.0
Random 1   period: 2        var   0.0
Random 1   period: 3        var   0.0
Random 1   period: 4        var   0.0
Random 2   formulation: 1   var   0.0
Random 2   formulation: 2   var   0.0
Repeated   Var              var   0.177845
=#

#EXAMPLE 5
################################################################################
################################################################################
################################################################################
#=
PROC MIXED data=df0;
CLASSES subject sequence period formulation;
MODEL  var = sequence period formulation/ DDFM=SATTERTH s;
RANDOM  formulation/TYPE=UN(1) SUB=subject G V;
RUN;

REML: 16.24111264
=#

lmm = LMM(@formula(var ~ sequence + period + formulation), df;
    random  = VarEffect(@covstr(formulation), VC),
    subject = :subject)
fit!(lmm)

#=
Linear Mixed Model: var ~ sequence + period + formulation
Random 1:
   Model: formulation
   Type: VarianceComponents (2)
   Coefnames: ["formulation: 1", "formulation: 2"]
Repeated:
   Model: nothing
   Type: ScaledIdentity (1)
   Coefnames: -

Status: converged

   -2 logREML: 16.2411

   Fixed effects:

Name             Estimate     SE
(Intercept)      1.61         0.334718
sequence: 2      -0.170833    0.277378
period: 2        0.144167     0.289463
period: 3        0.08         0.117047
period: 4        0.144167     0.289463
formulation: 2   -0.0791667   0.277378

Random effects:

   θ vector: [0.447322, 0.367367, 0.185068]

Random 1   formulation: 1   var   0.200097
Random 1   formulation: 2   var   0.134959
Repeated   Var              var   0.0342502
=#
################################################################################
