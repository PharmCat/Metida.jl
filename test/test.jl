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
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm) ≈ 25.129480634331067 atol=1E-6
    #Rholink function
    Metida.fit!(lmm; rholinkf = :atan)
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
    @test Metida.dof_satter(lmm, [0, 0, 0, 0, 0, 1]) ≈ 5.81896814947982 atol=1E-2
    @test nobs(lmm) == 20
    @test Metida.thetalength(lmm) == 3
    @test Metida.rankx(lmm) == 6
    @test Metida.gmatrix(lmm, 1)[1,1] ≈ 0.20009722409534533 atol=1E-6
    @test Metida.rmatrix(lmm, 1)[1,1] ≈ 0.034249998413262636 atol=1E-6
    @test dof(lmm) == 7
    @test vcov(lmm)[1,1]              ≈ 0.11203611149231425 atol=1E-6
    @test stderror(lmm)[1]            ≈ 0.33471795812641164 atol=1E-6
    @test length(modelmatrix(lmm)) == 120
    @test isa(response(lmm), Vector)
    #AI like algo
    Metida.fit!(lmm; aifirst = true, init = Metida.theta(lmm))
    @test Metida.m2logreml(lmm) ≈ 16.241112644506067 atol=1E-6
    #Set user coding
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(1 + formulation|subject), Metida.CSH; coding = Dict(:formulation => StatsModels.DummyCoding())),
    )
    Metida.fit!(lmm)
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
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    repeated = Metida.VarEffect(Metida.@covstr(formulation|subject), Metida.DIAG),
    )
    Metida.fit!(lmm)
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
@testset "  Model: Noblock, different subjects, DIAG/DIAG            " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation|subject), Metida.DIAG),
    repeated = Metida.VarEffect(Metida.@covstr(formulation|subject&period), Metida.DIAG),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm) ≈ 15.921517816789876 atol=1E-4 #need check
end
@testset "  Model: CSH/DIAG                                          " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation|subject), Metida.CSH),
    repeated = Metida.VarEffect(Metida.@covstr(formulation|subject), Metida.DIAG),
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
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3;
    random = [Metida.VarEffect(Metida.@covstr(r1|subject), Metida.CS), Metida.VarEffect(Metida.@covstr(r2|subject), Metida.CS)],
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm)  ≈ 710.4250214813896 atol=1E-8
end
@testset "  Model: Noblock, different subjects, AR/SI                " begin
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3;
    random = Metida.VarEffect(Metida.@covstr(p|s2&factor), Metida.AR),
    repeated = Metida.VarEffect(Metida.@covstr(r2|s2), Metida.SI),
    )
    Metida.fit!(lmm)
    # was^ @test Metida.m2logreml(lmm)  ≈ 839.5453114200564 atol=1E-8
    @test Metida.m2logreml(lmm)  ≈ 794.4253377905591 atol=1E-8
end
#=
@testset "  Model: Noblock, different subjects, ARMA/SI              " begin
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3;
    random = Metida.VarEffect(Metida.@covstr(p|s2&factor), Metida.ARMA),
    repeated = Metida.VarEffect(Metida.@covstr(r1|s2), Metida.SI),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm)  ≈ 835.5433832876931 atol=1E-8
end
=#
@testset "  Model: Noblock, different subjects, ARH/SI               " begin
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3;
    random = Metida.VarEffect(Metida.@covstr(r1|s2&factor), Metida.ARH),
    repeated = Metida.VarEffect(Metida.@covstr(r2|s2), Metida.DIAG),
    )
    Metida.fit!(lmm)
    # was: @test Metida.m2logreml(lmm)  ≈ 820.0534442753676 atol=1E-8
    @test Metida.m2logreml(lmm)  ≈ 698.5597498682596 atol=1E-8
end
@testset "  Model: Noblock, *,  DIAG/SI                              " begin
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3;
    random = Metida.VarEffect(Metida.@covstr(r1 * r2|subject), Metida.DIAG)
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm)  ≈ 715.4330098488135 atol=1E-8
end
@testset "  Model: Noblock, &,  DIAG/SI                              " begin
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3;
    random = Metida.VarEffect(Metida.@covstr(r1&r2|subject), Metida.DIAG),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm)  ≈ 719.9413776641368 atol=1E-8
end
@testset "  Model: Noblock, +,  DIAG/SI                              " begin
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3;
    random = Metida.VarEffect(Metida.@covstr(1 + r1 + r2|subject), Metida.DIAG; coding = Dict(:r1 => DummyCoding(), :r2 => DummyCoding())),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm)  ≈ 713.0655862252027 atol=1E-8
end
################################################################################
#                                  Errors
################################################################################
@testset "  Errors                                                   " begin

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
