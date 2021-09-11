# Metida

using  Test, CSV, DataFrames, StatsModels, StatsBase, LinearAlgebra, CategoricalArrays

path    = dirname(@__FILE__)
include("testdata.jl")

@testset "  Publick API basic tests                                  " begin
    io = IOBuffer();
    transform!(df0, :formulation => categorical, renamecols=false)
    # Basic, no block
    df0.nosubj = ones(size(df0, 1))
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation|nosubj), Metida.DIAG),
    )

    # Basic show (before fitting)
    Base.show(io, lmm)
    Metida.fit!(lmm)
    Base.show(io, lmm)
    Base.show(io, lmm.data)
    Base.show(io, lmm.result)
    Base.show(io, lmm.covstr)
    Base.show(io, lmm.covstr.repeated.covtype)
    Base.show(io, lmm.log)
    #
    @test Metida.m2logreml(lmm) ≈ 25.129480634331067 atol=1E-6
    #Verbose
    Metida.fit!(lmm; verbose = 2, io = io)
    @test Metida.m2logreml(lmm) ≈ 25.129480634331067 atol=1E-6
    #Basic, Subject block
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation|subject), Metida.DIAG),
    )
    Metida.fit!(lmm; aifirst = true)
    @test Metida.m2logreml(lmm) ≈ 16.241112644506067 atol=1E-6

    anovatable = Metida.anova(lmm;  ddf = :contain) # NOT VALIDATED
    anovatable = Metida.anova(lmm;  ddf = :residual)
    anovatable = Metida.anova(lmm)
    Base.show(io, anovatable)


    ############################################################################

    ############################################################################
    # API test
    ############################################################################
    l = [0 0 1 0 0 0; 0 0 0 1 0 0; 0 0 0 0 1 0]
    @test Metida.logreml(lmm)   ≈ -8.120556322253035 atol=1E-6
    @test isfitted(lmm) == true
    @test bic(lmm)              ≈ 24.558878811225412 atol=1E-6
    @test aic(lmm)              ≈ 22.241112644506067 atol=1E-6
    @test aicc(lmm)             ≈ 24.241112644506067 atol=1E-6
    @test Metida.caic(lmm)      ≈ 27.558878811225412 atol=1E-6
    @test dof_residual(lmm) == 14

    @test Metida.dof_satter(lmm, 6)   ≈ 5.81896814947982 atol=1E-2
    @test Metida.dof_satter(lmm)[end] ≈ 5.81896814947982 atol=1E-2
    @test Metida.dof_satter(lmm, [0 0 0 0 0 1]) ≈ 5.81896814947982 atol=1E-2
    @test Metida.dof_satter(lmm, l) ≈ 7.575447546211385 atol=1E-2
    @test Metida.fvalue(lmm, l) ≈  0.202727915619993 atol=1E-2
    @test Metida.dof_satter(lmm, Metida.lcontrast(lmm,3)) ≈ 7.575447546211385 atol=1E-2
    @test nobs(lmm) == 20
    @test Metida.thetalength(lmm) == 3
    @test Metida.rankx(lmm) == 6
    @test sum(Metida.gmatrix(lmm, 1)) ≈ 0.3350555603325126 atol=1E-6
    @test sum(Metida.rmatrix(lmm, 1)) ≈ 0.13699999248885292 atol=1E-6
    @test sum(Metida.vmatrix(lmm, 1)) ≈ 1.4772222338189034 atol=1E-6
    @test dof(lmm) == 7
    @test vcov(lmm)[1,1]              ≈ 0.11203611149231425 atol=1E-6
    @test stderror(lmm)[1]            ≈ 0.33471795812641164 atol=1E-6
    @test length(modelmatrix(lmm)) == 120
    @test isa(response(lmm), Vector)
    @test sum(Metida.hessian(lmm))    ≈ 1118.160713481362 atol=1E-2
    @test Metida.nblocks(lmm) == 5
    @test length(coefnames(lmm)) == 6
    @test Metida.confint(lmm)[end][1] ≈ -0.7630380758015894 atol=1E-4
    @test Metida.confint(lmm; ddf = :residual)[end][1] ≈ -0.6740837049617738 atol=1E-4

    Metida.confint(lmm; ddf = :contain)[end][1] #NOT VALIDATED
    @test size(crossmodelmatrix(lmm), 1) == 6
    @test anovatable.pval[4]          ≈ 0.7852154468081014 atol=1E-6
    ############################################################################
    # AI like algo
    Metida.fit!(lmm; aifirst = true, init = Metida.theta(lmm))
    @test Metida.m2logreml(lmm) ≈ 16.241112644506067 atol=1E-6
    # Score
    Metida.fit!(lmm; aifirst = :score)
    @test Metida.m2logreml(lmm) ≈ 16.241112644506067 atol=1E-6
    # AI
    Metida.fit!(lmm; aifirst = :ai)
    @test Metida.m2logreml(lmm) ≈ 16.241112644506067 atol=1E-6
    #Set user coding
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(1 + formulation|subject), Metida.CSH; coding = Dict(:formulation => StatsModels.DummyCoding())),
    )
    # Test varlink/rholinkf
    Metida.fit!(lmm; rholinkf = :sqsigm)
    @test Metida.dof_satter(lmm, [0, 0, 0, 0, 0, 1]) ≈ 6.043195705464293 atol=1E-2
    @test Metida.m2logreml(lmm) ≈ 10.314822559210157 atol=1E-6
    @test_nowarn Metida.fit!(lmm; varlinkf = :sq)
    # Repeated effect only
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    repeated = Metida.VarEffect(Metida.@covstr(formulation|nosubj)),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm) ≈ 25.129480634331063 atol=1E-6

    #BE like
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation|subject), Metida.CSH),
    repeated = Metida.VarEffect(Metida.@covstr(formulation|subject), Metida.DIAG),
    )
    Metida.fit!(lmm; aifirst = :score)
    @test Metida.m2logreml(lmm) ≈ 10.065238626765524 atol=1E-6
    #incomplete
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df1;
    random = Metida.VarEffect(Metida.@covstr(formulation|subject), Metida.CSH),
    repeated = Metida.VarEffect(Metida.@covstr(formulation|subject), Metida.DIAG),
    )
    Metida.fit!(lmm; hes = false)
    @test Metida.m2logreml(lmm) ≈ 14.819463206995163 atol=1E-6
    @test Metida.dof_satter(lmm, 6)   ≈ 3.981102548214154 atol=1E-2

    lmm = Metida.LMM(@formula(var~period*formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation+sequence|nosubj), Metida.SI),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm, [0.222283, 0.444566]) ≈ Metida.m2logreml(lmm) atol=1E-6
    # EXPERIMENTAL
    @test Metida.dof_contain(lmm, 1) == 12
    @test Metida.dof_contain(lmm, 5) == 8
end
################################################################################
#                                  df0
################################################################################
@testset "  Model: Only repeated, 0/DIAG                             " begin
    # nowarn
    io = IOBuffer();
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    repeated = Metida.VarEffect(Metida.@covstr(formulation|subject), Metida.DIAG),
    )
    Metida.fit!(lmm)
    Base.show(io, lmm)
    @test Metida.m2logreml(lmm) ≈ 25.000777869122338 atol=1E-8
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    repeated = Metida.VarEffect(Metida.@covstr(formulation|subject), Metida.DIAG)
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm) ≈ 25.000777869122338 atol=1E-8
end
@testset "  Model: Only repeated, noblock, 0/CSH (rholinkf = :atan)  " begin
    # nowarn
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    repeated = Metida.VarEffect(Metida.@covstr(period|subject), Metida.CSH),
    )
    Metida.fit!(lmm; rholinkf = :atan)
    @test Metida.m2logreml(lmm) ≈ 8.740095378772942 atol=1E-8
end
@testset "  Model: Only random, noblock, SI/SI                       " begin
    # nowarn
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(subject|nosubj), Metida.SI),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm) ≈ 10.862124583312674 atol=1E-8
end
@testset "  Model: Only random, INT, SI/SI                           " begin
    # nowarn
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(1|subject), Metida.SI),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm) ≈ 10.862124583312674 atol=1E-8
end
@testset "  Model: Noblock, equal subjects, CSH/CS                   " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation|subject), Metida.CSH),
    repeated = Metida.VarEffect(Metida.@covstr(formulation|subject), Metida.CS),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm) ≈ 10.3039977509049 atol=1E-6 #need check
end
@testset "  Model: Different subjects, INT, CSH/DIAG                 " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(1 + formulation|subject), Metida.CSH; coding = Dict(:formulation => DummyCoding())),
    repeated = Metida.VarEffect(Metida.@covstr(formulation|subject&period), Metida.DIAG),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm) ≈ 10.06523870216023 atol=1E-4 #need check
end
@testset "  Model: CSH/DIAG (rholinkf = :psigm)                      " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation|subject), Metida.CSH),
    repeated = Metida.VarEffect(Metida.@covstr(formulation|subject), Metida.DIAG),
    )
    Metida.fit!(lmm; rholinkf = :psigm)
    @test Metida.m2logreml(lmm) ≈ 10.065239006121315 atol=1E-6
end
@testset "  Model: Custom covariance type                            " begin
    CCTG = Metida.CovarianceType(Metida.CovmatMethod((q,p) -> (q, 1), Metida.gmat_csh!))
    CCTR = Metida.CovarianceType(Metida.CovmatMethod((q,p) -> (q, 0), Metida.rmatp_diag!))
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation|subject), CCTG),
    repeated = Metida.VarEffect(Metida.@covstr(formulation|subject), CCTR),
    )

    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm) ≈ 10.065239006121315 atol=1E-6
end
################################################################################
#                                  ftdf / 1fptime.csv
################################################################################
@testset "  Model: Categorical * Continuous predictor, CSH/SI        " begin
    # nowarn
    lmm = Metida.LMM(@formula(response ~1 + factor*time), ftdf;
    random = Metida.VarEffect(Metida.@covstr(1 + time|subject&factor), Metida.CSH),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm) ≈ 1300.1807598168923 atol=1E-6
    @test coef(lmm) ≈ [22.13309710783416, 2.000486297455917, 1.1185284725578566, 0.4049714576872601] atol=1E-6
    @test Metida.dof_satter(lmm, [0, 0, 0, 1]) ≈ 37.999999999991786 atol=1E-2

end
@testset "  Model: Function terms, CSH/SI                            " begin
    ftdf.expresp = exp.(ftdf.response)
    ftdf.exptime = exp.(ftdf.time)
    lmm = Metida.LMM(@formula(log(expresp) ~ 1 + factor*log(exptime)), ftdf;
    random = Metida.VarEffect(Metida.@covstr(1 + log(exptime)|subject&factor), Metida.CSH),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm) ≈ 1300.1807598168923 atol=1E-6
end
################################################################################
#                                  ftdf2 / 1freparma.csv
################################################################################
@testset "  Model: Categorical * Continuous predictor, 0/ARMA        " begin
    # nowarn
    # SPSS 715.452856
    io = IOBuffer();
    lmm = Metida.LMM(@formula(response ~ 1 + factor*time), ftdf2;
    repeated = Metida.VarEffect(Metida.@covstr(time|subject&factor), Metida.ARMA),
    )
    Metida.fit!(lmm)
    println(io, lmm.log)
    @test Metida.m2logreml(lmm) ≈ 715.4528559688382 atol = 1E-6
end
@testset "  Model: Categorical * Continuous predictor, DIAG/AR       " begin
    # nowarn
    lmm = Metida.LMM(@formula(response ~ 1 + factor*time), ftdf2;
    random = Metida.VarEffect(Metida.@covstr(factor|subject&factor), Metida.DIAG),
    repeated = Metida.VarEffect(Metida.@covstr(1|subject&factor), Metida.AR),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm) ≈ 710.0962305879676 atol=1E-6
end
@testset "  Model: Categorical * Continuous predictor, 0/ARH         " begin
    # nowarn
    lmm = Metida.LMM(@formula(response ~ 1 + factor*time), ftdf2;
    repeated = Metida.VarEffect(Metida.@covstr(factor|subject&factor), Metida.ARH),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm) ≈ 731.7794071577566 atol=1E-6
end
################################################################################
#                                  ftdf3 / 2f2rand.csv
################################################################################
@testset "  Model: CS, CS/SI                                         " begin
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3; contrasts=Dict(:factor => DummyCoding(; base=1.0)),
    random = [Metida.VarEffect(Metida.@covstr(r1|subject), Metida.CS), Metida.VarEffect(Metida.@covstr(r2|subject), Metida.CS)],
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm)  ≈ 710.4250214813896 atol=1E-8
end
@testset "  Model: AR/SI                                             " begin
    # SPSS 698.879
    # nowarn
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3; contrasts=Dict(:factor => DummyCoding(; base=1.0)),
    random = Metida.VarEffect(Metida.@covstr(r1|subject), Metida.AR),
    repeated = Metida.VarEffect(Metida.@covstr(p|subject), Metida.DIAG),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm)  ≈ 698.8792511057682 atol=1E-8
end

@testset "  Model: ARMA/SI                                           " begin
    io = IOBuffer();
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3; contrasts=Dict(:factor => DummyCoding(; base=1.0)),
    random = Metida.VarEffect(Metida.@covstr(p|r1&r2), Metida.ARMA),
    )
    Metida.fit!(lmm; verbose = 3, io = io)
    #[1.2964e-5, 0.0299594, 0.0699728, 3.69557]
    println(io, lmm.log)
    @test Metida.m2logreml(lmm)  ≈ 913.9176298311813 atol=1E-8
end

@testset "  Model: ARH/SI (subjects with &)                          " begin
    # SPSS 707.377
    # nowarn
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3; contrasts=Dict(:factor => DummyCoding(; base=1.0)),
    random = Metida.VarEffect(Metida.@covstr(r1|s2&factor), Metida.ARH),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm)  ≈ 707.3765873864152 atol=1E-8
    @test Metida.dof_satter(lmm, [0, 1]) ≈ 23.111983305626193 atol=1E-2
end
@testset "  Model: INT, *, DIAG/SI                                   " begin
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3;
    random = Metida.VarEffect(Metida.@covstr(1 + r2 * r1|subject), Metida.DIAG; coding=Dict(:r1 => DummyCoding(), :r2 => DummyCoding()))
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm)  ≈ 713.0655862252027 atol=1E-8
end
@testset "  Model: &, DIAG/SI                                        " begin
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3;
    random = Metida.VarEffect(Metida.@covstr(r1&r2|subject), Metida.DIAG),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm)  ≈ 719.9413776641368 atol=1E-8
end
@testset "  Model: INT, +,  TOEPHP(3)/SI                             " begin
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3;
    random = Metida.VarEffect(Metida.@covstr(1 + r1 + r2|subject), Metida.TOEPHP(3); coding = Dict(:r1 => DummyCoding(), :r2 => DummyCoding())),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm)  ≈ 705.9946274598822 atol=1E-8
end
@testset "  Model: TOEP/SI                                           " begin
    # SPSS 710.200
    # nowarn
    io = IOBuffer();
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3;
    random = Metida.VarEffect(Metida.@covstr(r1|subject), Metida.TOEP),
    )
    Metida.fit!(lmm)
    Base.show(io, lmm)
    @test Metida.m2logreml(lmm)  ≈ 710.1998669150806 atol=1E-8
end
@testset "  Model: TOEPP(2)/SI                                       " begin
    # nowarn
    io = IOBuffer();
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3;
    random = Metida.VarEffect(Metida.@covstr(r1|subject), Metida.TOEPP(2)),
    )
    Metida.fit!(lmm)
    Base.show(io, lmm)
    @test Metida.m2logreml(lmm)  ≈ 715.2410264030134 atol=1E-8
end
@testset "  Model: DIAG/TOEPP(3)                                     " begin
    # nowarn
    io = IOBuffer();
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3;
    random = Metida.VarEffect(Metida.@covstr(r2|subject), Metida.DIAG),
    repeated = Metida.VarEffect(Metida.@covstr(p|subject), Metida.TOEPP(3)),
    )
    Metida.fit!(lmm)
    Base.show(io, lmm)
    @test Metida.m2logreml(lmm)  ≈ 773.9575538254085 atol=1E-8
end
@testset "  Model: TOEPH/SI                                          " begin
    # nowarn
    io = IOBuffer();
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3;
    random = Metida.VarEffect(Metida.@covstr(r1|subject), Metida.TOEPH),
    )
    Metida.fit!(lmm)
    Base.show(io, lmm)
    @test Metida.m2logreml(lmm)  ≈ 705.7916833009426 atol=1E-8
end
@testset "  Model: SI/TOEPHP(3)                                      " begin
    io = IOBuffer();
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3;
    random = Metida.VarEffect(Metida.@covstr(r1|subject), Metida.SI),
    repeated = Metida.VarEffect(Metida.@covstr(r1&r2|subject), Metida.TOEPHP(3)),
    )
    Metida.fit!(lmm; optmethod = Metida.LBFGS_OM)
    Base.show(io, lmm)
    @test Metida.m2logreml(lmm)  ≈ 713.5850978377632 atol=1E-8
end
@testset "  Model: BE RDS 1, FDA model                               "  begin
    dfrds        = CSV.File(joinpath(path, "csv", "berds", "rds1.csv"), types = Dict(:PK => Float64, :subject => String, :period => String, :sequence => String, :treatment => String )) |> DataFrame
    dropmissing!(dfrds)
    dfrds.lnpk = log.(dfrds.PK)
    lmm = Metida.LMM(@formula(lnpk~sequence+period+treatment), dfrds;
    random = Metida.VarEffect(Metida.@covstr(treatment|subject), Metida.CSH),
    repeated = Metida.VarEffect(Metida.@covstr(treatment|subject), Metida.DIAG),
    )
    Metida.fit!(lmm)
    @test collect(Metida.confint(lmm)[6]) ≈  [0.05379033790060175, 0.23713821749515449] atol=1E-8
    anovatable = Metida.anova(lmm)
    @test anovatable.pval ≈ [3.087934998046721e-63, 0.9176105002577626, 0.6522549061162943, 0.002010933915677479] atol=1E-4

    lmm = Metida.LMM(@formula(lnpk~0+sequence+period+treatment), dfrds;
    random = Metida.VarEffect(Metida.@covstr(treatment|subject), Metida.CSH),
    repeated = Metida.VarEffect(Metida.@covstr(treatment|subject), Metida.DIAG),
    )
    Metida.fit!(lmm)
    anovatable = Metida.anova(lmm)
        @test anovatable.pval ≈ [8.129457925585042e-74, 0.6522549061174356, 0.0020109339157131302] atol=1E-4
end
################################################################################
#                                  Errors
################################################################################
@testset "  Errors test                                              " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation|nosubj), Metida.DIAG),
    )
    @test_throws ErrorException Metida.fit!(lmm; init = [1.0])
    @test_throws ErrorException Metida.hessian(lmm)
    @test_throws ErrorException Metida.dof_satter(lmm)
    @test_throws ErrorException Metida.confint(lmm)

    @test_throws ErrorException  Metida.LMM(@formula(var~sequence+period+formulation), df0;)

    @test_throws ErrorException  Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = [Metida.VarEffect(Metida.@covstr(formulation|nosubj), Metida.DIAG), Metida.VarEffect(Metida.@covstr(formulation|nosubj), Metida.RZero())]
    )
    # Error messages
    io = IOBuffer();
    lmm = Metida.LMM(@formula(response ~ 1 + factor*time), ftdf2;
    random = Metida.VarEffect(Metida.@covstr(factor|subject&factor), Metida.DIAG),
    repeated = Metida.VarEffect(Metida.@covstr(1|subject&factor), Metida.ARMA),
    )
    Metida.fit!(lmm)
    println(io, lmm.log)
end
################################################################################
#                                  Sweep test
################################################################################
@testset "  Sweep operator test                                      " begin
    A =
[1.0  2  2  4  1
 2  2  3  3  5
 2  3  3  4  2
 4  3  4  4  5
 1  5  2  5  5]
    iA =  inv(A[1:4, 1:4])
    iAs = Symmetric(-Metida.sweep!(copy(A), 1:4; syrkblas = true)[1][1:4, 1:4])
    B = copy(A)
    for i = 1:4
        Metida.sweep!(B, i; syrkblas = false)
    end
    iAss = Symmetric(-B[1:4, 1:4])
    akk = zeros(5)
    iAb = Symmetric(-Metida.sweepb!(view(akk, 1:5), copy(A), 1:4)[1][1:4, 1:4])
    @test iA  ≈ iAs  atol=1E-6
    @test iA  ≈ iAss atol=1E-6
    @test iAs ≈ iAb  atol=1E-6
end

@testset "  Experimental                                             " begin
    lmm = Metida.LMM(@formula(response ~ 1), ftdf;
    repeated = Metida.VarEffect(Metida.@covstr(response+time|subject), Metida.SPEXP),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm) ≈ 1528.7150702624508 atol=1E-6

    @test_nowarn Metida.fit!(lmm; varlinkf = :identity)
end
