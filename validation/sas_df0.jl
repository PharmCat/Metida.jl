################################################################################
################################################################################
################################################################################
#=
PROC MIXED data=df0;
CLASSES subject sequence period formulation;
MODEL  var = sequence period formulation/ DDFM=SATTERTH LComponents E solution covb covbi;
CONTRAST 'CONTR' formulation    1 -1;
RANDOM  formulation/TYPE=CSH SUB=subject G V;
REPEATED/GRP=formulation SUB=subject R;
ESTIMATE 'T vs. R' formulation 1 -1/CL ALPHA=0.1 E;
RUN;

REML: 10.06523862
=#
################################################################################
################################################################################
################################################################################
#=
PROC MIXED data=df0;
CLASSES subject sequence period formulation;
MODEL  var = sequence period formulation/ DDFM=SATTERTH LComponents E solution covb covbi;
CONTRAST 'CONTR' formulation    1 -1;
RANDOM  formulation/TYPE=VC SUB=subject G V;
REPEATED/GRP=formulation SUB=subject R;
ESTIMATE 'T vs. R' formulation 1 -1/CL ALPHA=0.1 E;
RUN;

REML: 16.06148160
=#


lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
           random = Metida.VarEffect(Metida.@covstr(formulation), Metida.SI),
           repeated = Metida.VarEffect(Metida.@covstr(formulation), Metida.VC),
           subject = :subject
           )
Metida.fit!(lmm)
#=
julia> Metida.fit!(lmm)
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


################################################################################
################################################################################
################################################################################

PROC MIXED data=df0;
CLASSES subject sequence period formulation;
MODEL  var = sequence period formulation/ DDFM=SATTERTH LComponents E solution covb covbi;
CONTRAST 'CONTR' formulation    1 -1;
RANDOM  subject/TYPE=VC G V;
ESTIMATE 'T vs. R' formulation 1 -1/CL ALPHA=0.1 E;
RUN;

REML: 10.86212458

################################################################################
################################################################################
################################################################################

PROC MIXED data=df0;
CLASSES subject sequence period formulation;
MODEL  var = sequence period formulation/ DDFM=SATTERTH LComponents E solution covb covbi;
CONTRAST 'CONTR' formulation    1 -1;
RANDOM  period/TYPE=VC G V;
RANDOM  formulation/TYPE=VC G V;
ESTIMATE 'T vs. R' formulation 1 -1/CL ALPHA=0.1 E;
RUN;

REML: 25.12948063

################################################################################
################################################################################
################################################################################

PROC MIXED data=df0;
CLASSES subject sequence period formulation;
MODEL  var = sequence period formulation/ DDFM=SATTERTH LComponents E solution covb covbi;
CONTRAST 'CONTR' formulation    1 -1;
RANDOM  formulation/TYPE=UN(1) SUB=subject G V;
ESTIMATE 'T vs. R' formulation 1 -1/CL ALPHA=0.1 E;
RUN;

REML: 16.24111264

################################################################################
