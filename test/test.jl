# Metida

using  Test, CSV, DataFrames, StatsModels, StatsBase, LinearAlgebra

path    = dirname(@__FILE__)
include("testdata.jl")

@testset "  Basic test                                               " begin
    io = IOBuffer();
    #Basic, no block
    df0.nosubj = ones(size(df0, 1))
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation|nosubj), Metida.DIAG),
    )
    Base.show(io, lmm)
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm) ≈ 25.129480634331067 atol=1E-6
    #Rholink function
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm) ≈ 25.129480634331063 atol=1E-6 #need chec
    #Verbose
    Metida.fit!(lmm; verbose = 2, io = io)
    @test Metida.m2logreml(lmm) ≈ 25.129480634331067 atol=1E-6
    #Basic, Subject block
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation|subject), Metida.DIAG),
    )
    Metida.fit!(lmm; aifirst = true)
    @test Metida.m2logreml(lmm) ≈ 16.241112644506067 atol=1E-6
    #API test
    @test Metida.logreml(lmm)   ≈ -8.120556322253035 atol=1E-6
    @test isfitted(lmm) == true
    @test bic(lmm)              ≈ 24.558878811225412 atol=1E-6
    @test aic(lmm)              ≈ 22.241112644506067 atol=1E-6
    @test aicc(lmm)             ≈ 24.241112644506067 atol=1E-6
    @test Metida.caic(lmm)      ≈ 27.558878811225412 atol=1E-6
    @test dof_residual(lmm) == 14
    @test Metida.dof_contain(lmm) == 6
    @test Metida.dof_satter(lmm, [0, 0, 0, 0, 0, 1]) ≈ 5.81896814947982 atol=1E-2
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
    #AI like algo
    Metida.fit!(lmm; aifirst = true, init = Metida.theta(lmm))
    @test Metida.m2logreml(lmm) ≈ 16.241112644506067 atol=1E-6
    #Set user coding
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(1 + formulation|subject), Metida.CSH; coding = Dict(:formulation => StatsModels.DummyCoding())),
    )
    Metida.fit!(lmm; rholinkf = :sqsigm)
    @test Metida.dof_satter(lmm, [0, 0, 0, 0, 0, 1]) ≈ 6.043195705464293 atol=1E-2
    @test Metida.m2logreml(lmm) ≈ 10.314822559210157 atol=1E-6
    #Repeated only
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    repeated = Metida.VarEffect(Metida.@covstr(formulation|nosubj)),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm) ≈ 25.129480634331063 atol=1E-6
    Base.show(io, lmm)
    Base.show(io, lmm.data)
    Base.show(io, lmm.result)
    Base.show(io, lmm.covstr)
    Base.show(io, lmm.covstr.repeated.covtype)
    Base.show(io, lmm.log)
end
################################################################################
#                                  df0
################################################################################
@testset "  Model: Only repeated 0/DIAG                              " begin
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
@testset "  Model: Only repeated, noblock 0/CSH (rholinkf = :atan)   " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    repeated = Metida.VarEffect(Metida.@covstr(period|subject), Metida.CSH),
    )
    Metida.fit!(lmm; rholinkf = :atan)
    @test Metida.m2logreml(lmm) ≈ 8.740095378772942 atol=1E-8
end
@testset "  Model: Only random, noblock SI/SI                        " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(subject|nosubj), Metida.SI),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm) ≈ 10.862124583312674 atol=1E-8
end
@testset "  Model: Only random, Int SI/SI                            " begin
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
@testset "  Model: different subjects, random int,  CSH/DIAG         " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(1 + formulation|subject), Metida.CSH; coding = Dict(:formulation => DummyCoding())),
    repeated = Metida.VarEffect(Metida.@covstr(formulation|subject&period), Metida.DIAG),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm) ≈ 10.06523870216023 atol=1E-4 #need check
end
@testset "  Model: CSH/DIAG                                          " begin
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
#                                  ftdf
################################################################################
@testset "  Model: Categorical * Continuous predictor CSH/SI         " begin
    lmm = Metida.LMM(@formula(response ~1 + factor*time), ftdf;
    random = Metida.VarEffect(Metida.@covstr(1 + time|subject&factor), Metida.CSH),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm) ≈ 1300.1807598168923 atol=1E-6
    @test coef(lmm) ≈ [22.13309710783416, 2.000486297455917, 1.1185284725578566, 0.4049714576872601] atol=1E-6
    @test Metida.dof_satter(lmm, [0, 0, 0, 1]) ≈ 37.999999999991786 atol=1E-2

end
@testset "  Model: Function terms CSH/SI                             " begin
    ftdf.expresp = exp.(ftdf.response)
    ftdf.exptime = exp.(ftdf.time)
    lmm = Metida.LMM(@formula(log(expresp) ~ 1 + factor*log(exptime)), ftdf;
    random = Metida.VarEffect(Metida.@covstr(1 + log(exptime)|subject&factor), Metida.CSH),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm) ≈ 1300.1807598168923 atol=1E-6
end
################################################################################
#                                  ftdf2
################################################################################
@testset "  Model: Categorical * Continuous predictor DIAG/ARMA      " begin
    lmm = Metida.LMM(@formula(response ~ 1 + factor*time), ftdf2;
    random = Metida.VarEffect(Metida.@covstr(factor|subject&factor), Metida.DIAG),
    repeated = Metida.VarEffect(Metida.@covstr(1|subject&factor), Metida.ARMA),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm) ≈ 709.1400046571733 atol=1E-6
end
@testset "  Model: Categorical * Continuous predictor DIAG/AR        " begin
    lmm = Metida.LMM(@formula(response ~ 1 + factor*time), ftdf2;
    random = Metida.VarEffect(Metida.@covstr(factor|subject&factor), Metida.DIAG),
    repeated = Metida.VarEffect(Metida.@covstr(1|subject&factor), Metida.AR),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm) ≈ 710.0962305879676 atol=1E-6
end
@testset "  Model: Categorical * Continuous predictor 0/ARH          " begin
    lmm = Metida.LMM(@formula(response ~ 1 + factor*time), ftdf2;
    repeated = Metida.VarEffect(Metida.@covstr(factor|subject&factor), Metida.ARH),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm) ≈ 731.7794071577566 atol=1E-6
end
################################################################################
#                                  ftdf3
################################################################################
@testset "  Model: CS,CS/SI                                          " begin
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3; contrasts=Dict(:factor => DummyCoding(; base=1.0)),
    random = [Metida.VarEffect(Metida.@covstr(r1|subject), Metida.CS), Metida.VarEffect(Metida.@covstr(r2|subject), Metida.CS)],
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm)  ≈ 710.4250214813896 atol=1E-8
end
@testset "  Model: Noblock, different subjects, AR/SI                " begin
    #SPSS 698.879
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3; contrasts=Dict(:factor => DummyCoding(; base=1.0)),
    random = Metida.VarEffect(Metida.@covstr(r1|subject), Metida.AR),
    repeated = Metida.VarEffect(Metida.@covstr(p|subject), Metida.DIAG),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm)  ≈ 698.8792511057682 atol=1E-8
end

@testset "  Model: Noblock, different subjects, ARMA/SI              " begin
    io = IOBuffer();
    #SPSS 904.236!!! random = Metida.VarEffect(Metida.@covstr(s2|r1&r2), Metida.ARMA),
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3; contrasts=Dict(:factor => DummyCoding(; base=1.0)),
    random = Metida.VarEffect(Metida.@covstr(p|r1&r2), Metida.ARMA),
    )
    Metida.fit!(lmm; verbose = 3, io = io)
    println(io, lmm.log)
    @test Metida.m2logreml(lmm)  ≈ 913.9176298311813 atol=1E-8
end

@testset "  Model: Noblock, different subjects, ARH/SI               " begin
    #SPSS 707.377
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3; contrasts=Dict(:factor => DummyCoding(; base=1.0)),
    random = Metida.VarEffect(Metida.@covstr(r1|s2&factor), Metida.ARH),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm)  ≈ 707.3765873864152 atol=1E-8
end
@testset "  Model: Noblock, *,  DIAG/SI                              " begin
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3;
    random = Metida.VarEffect(Metida.@covstr(r1 * r2|subject), Metida.DIAG)
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm)  ≈ 715.4330098488135 atol=1E-8
    @test Metida.dof_satter(lmm, [0, 1]) ≈ 14.344012741005523 atol=1E-2
end
@testset "  Model: Noblock, &,  DIAG/SI                              " begin
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3;
    random = Metida.VarEffect(Metida.@covstr(r1&r2|subject), Metida.DIAG),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm)  ≈ 719.9413776641368 atol=1E-8
end
@testset "  Model: Noblock, +,  TOEPHP/SI                            " begin
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3;
    random = Metida.VarEffect(Metida.@covstr(1 + r1 + r2|subject), Metida.TOEPHP(3); coding = Dict(:r1 => DummyCoding(), :r2 => DummyCoding())),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm)  ≈ 705.9946274598822 atol=1E-8
end
@testset "  Model:  TOEP/SI                                          " begin
    #SPSS 710.200
    io = IOBuffer();
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3;
    random = Metida.VarEffect(Metida.@covstr(r1|subject), Metida.TOEP),
    )
    Metida.fit!(lmm)
    Base.show(io, lmm)
    @test Metida.m2logreml(lmm)  ≈ 710.1998669150806 atol=1E-8
end
@testset "  Model:  TOEPP/SI                                         " begin
    io = IOBuffer();
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3;
    random = Metida.VarEffect(Metida.@covstr(r1|subject), Metida.TOEPP(2)),
    )
    Metida.fit!(lmm)
    Base.show(io, lmm)
    @test Metida.m2logreml(lmm)  ≈ 715.2410264030134 atol=1E-8
end
@testset "  Model:  DIAG/TOEPP                                       " begin
    io = IOBuffer();
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3;
    random = Metida.VarEffect(Metida.@covstr(r2|subject), Metida.DIAG),
    repeated = Metida.VarEffect(Metida.@covstr(p|subject), Metida.TOEPP(3)),
    )
    Metida.fit!(lmm)
    Base.show(io, lmm)
    @test Metida.m2logreml(lmm)  ≈ 773.9575538254085 atol=1E-8
end
@testset "  Model:  TOEPH/SI                                         " begin
    io = IOBuffer();
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3;
    random = Metida.VarEffect(Metida.@covstr(r1|subject), Metida.TOEPH),
    )
    Metida.fit!(lmm)
    Base.show(io, lmm)
    @test Metida.m2logreml(lmm)  ≈ 705.7916833009426 atol=1E-8
end
@testset "  Model:  SI/TOEPHP                                        " begin
    io = IOBuffer();
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3;
    random = Metida.VarEffect(Metida.@covstr(r1|subject), Metida.SI),
    repeated = Metida.VarEffect(Metida.@covstr(r1&r2|subject), Metida.TOEPHP(3)),
    )
    Metida.fit!(lmm)
    Base.show(io, lmm)
    @test Metida.m2logreml(lmm)  ≈ 713.5850978377632 atol=1E-8
end
################################################################################
#                                  Errors
################################################################################
@testset "  Errors                                                   " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation|nosubj), Metida.DIAG),
    )
    @test_throws ErrorException Metida.fit!(lmm; init = [1.0])
    @test_throws ErrorException Metida.hessian(lmm)

    @test_throws ErrorException  Metida.LMM(@formula(var~sequence+period+formulation), df0;)

    @test_throws ErrorException  Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = [Metida.VarEffect(Metida.@covstr(formulation|nosubj), Metida.DIAG), Metida.VarEffect(Metida.@covstr(formulation|nosubj), Metida.RZero())]
    )
end
################################################################################
#                                  Sweep test
################################################################################
@testset "  Sweep test                                               " begin
    A =
[1.0  2  2  4  1
 2  2  3  3  5
 2  3  3  4  2
 4  3  4  4  5
 1  5  2  5  5]
    iA =  inv(A[1:4, 1:4])
    iAs = Symmetric(-Metida.sweep!(copy(A), 1:4; syrkblas = true)[1:4, 1:4])
    B = copy(A)
    for i = 1:4
        Metida.sweep!(B, i; syrkblas = false)
    end
    iAss = Symmetric(-B[1:4, 1:4])
    akk = zeros(5)
    iAb = Symmetric(-Metida.sweepb!(view(akk, 1:5), copy(A), 1:4)[1:4, 1:4])
    @test iA  ≈ iAs atol=1E-6
    @test iA  ≈ iAss atol=1E-6
    @test iAs ≈ iAb atol=1E-6
end
