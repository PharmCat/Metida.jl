# Metida

using  Test, CSV, DataFrames, StatsModels, StatsBase, LinearAlgebra

path    = dirname(@__FILE__)
include("testdata.jl")

@testset "  Basic test                                               " begin
    io = IOBuffer();
    #Basic, no block
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation), Metida.DIAG),
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
    random = Metida.VarEffect(Metida.@covstr(formulation), Metida.DIAG),
    subject = :subject)
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
    random = Metida.VarEffect(Metida.@covstr(1 + formulation), Metida.CSH; coding = Dict(:formulation => StatsModels.DummyCoding())),
    subject = :subject)
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm) ≈ 10.314822559210157 atol=1E-6
    #Repeated only
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    repeated = Metida.VarEffect(Metida.@covstr(formulation)),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm) ≈ 25.129480634331063 atol=1E-6
    Base.show(io, lmm)
    Base.show(io, lmm.data)
    Base.show(io, lmm.result)
    Base.show(io, lmm.covstr)
    Base.show(io, lmm.log)
end
@testset "  Model: Categorical * Continuous predictor                " begin
    lmm = Metida.LMM(@formula(response ~1 + factor*time), ftdf;
    random = Metida.VarEffect(Metida.@covstr(1 + time), Metida.CSH),
    subject = [:subject, :factor]
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm) ≈ 1300.1807598168923 atol=1E-6
end
@testset "  Model: Function terms                                    " begin
    ftdf.expresp = exp.(ftdf.response)
    ftdf.exptime = exp.(ftdf.time)
    lmm = Metida.LMM(@formula(log(expresp) ~ 1 + factor*log(exptime)), ftdf;
    random = Metida.VarEffect(Metida.@covstr(1 + log(exptime)), Metida.CSH),
    subject = [:subject, :factor]
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm) ≈ 1300.1807598168923 atol=1E-6
end
@testset "  Model: Only random (SI), noblock                         " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(subject), Metida.SI),
    )
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 10.862124583312674 atol=1E-8
end
@testset "  Model: Only repeated (CSH), noblock                      " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    repeated = Metida.VarEffect(Metida.@covstr(period), Metida.CSH),
    subject = :subject
    )
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 8.740095378772942 atol=1E-8
end

@testset "  Model: CSH/subject + DIAG                                " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation), Metida.CSH),
    repeated = Metida.VarEffect(Metida.@covstr(formulation), Metida.DIAG),
    subject = :subject
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm) ≈ 10.065239006121315 atol=1E-6
end
@testset "  Model: DIAG/subject + nothing                            " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation), Metida.DIAG),
    subject = :subject
    )
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 16.241112644506067 atol=1E-8

end

@testset "  Model: CSH + nothing                                     " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation), Metida.CSH),
    )
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 25.129480634331067 atol=1E-8
end
@testset "  Model: (DIAG(period) + DIAG(formulation)) + nothing      " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = [Metida.VarEffect(Metida.@covstr(period), Metida.DIAG), Metida.VarEffect(Metida.@covstr(formulation), Metida.DIAG)],
    )
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 25.12948063433108 atol=1E-8

end
@testset "  Model: CSH + CSH                                         " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation), Metida.CSH, subj = :subject),
    repeated = Metida.VarEffect(Metida.@covstr(formulation), Metida.CSH, subj = :subject),
    )
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 9.948203272948428 atol=1E-6 #need check
end
@testset "  Model: CSH/subject + nothing                             " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation), Metida.CSH),
    subject = :subject
    )
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 10.314822655850815 atol=1E-6
end
@testset "  Model: SI/subject + DIAG                                 " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation), Metida.SI),
    repeated = Metida.VarEffect(Metida.@covstr(formulation), Metida.DIAG),
    subject = :subject
    )
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 16.061481604620425 atol=1E-8

end
@testset "  Model: CSH/subject + CSH                                 " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation), Metida.CSH),
    repeated = Metida.VarEffect(Metida.@covstr(formulation), Metida.CSH),
    subject = :subject
    )
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 9.948203272948428 atol=1E-6
end
@testset "  Model: (CSH+DIAG) + nothing                              " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = [Metida.VarEffect(Metida.@covstr(formulation), Metida.CSH),
    Metida.VarEffect(Metida.@covstr(period), Metida.DIAG)],
    )
    Metida.fit!(lmm)
    @test true
end
@testset "  Model: CSH(formulation + period) + nothing               " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation + period), Metida.CSH),
    )
    Metida.fit!(lmm)
    @test true
end
@testset "  Model: CSH(formulation & period) + nothing               " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation & period), Metida.CSH), subject = :subject
    )
    Metida.fit!(lmm)
    @test true
end
@testset "  Model: CSH(formulation * period) + nothing               " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation * period), Metida.DIAG), subject = :subject
    )
    Metida.fit!(lmm)
    #@test lmm.result.reml ≈ 13.555817544390917 atol=1E-4
    @test true
end

@testset "  Model: DIAG + DIAG                                       " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation), Metida.DIAG, subj = :subject),
    repeated = Metida.VarEffect(Metida.@covstr(formulation), Metida.DIAG, subj = [:subject, :period]),
    )
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 15.921517816789876 atol=1E-4 #need check
end

@testset "  Model: DIAG + DIAG                                       " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation), Metida.DIAG, subj = :subject),
    repeated = Metida.VarEffect(Metida.@covstr(formulation), Metida.DIAG, subj = [:subject, :period]),
    subject = :sequence)
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 25.00077786912234 atol=1E-4 #need check
end

@testset "  Model: DIAG + DIAG                                       " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation), Metida.DIAG, subj = :subject),
    repeated = Metida.VarEffect(Metida.@covstr(formulation), Metida.DIAG, subj = [:period]),
    subject = [:sequence, :formulation])
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 25.00077786912234 atol=1E-4 #need check
end

@testset "  Errors                                                   " begin
    @test_throws ArgumentError Metida.VarEffect(Metida.@covstr(formulation), Metida.DIAG, subj = "subj")
    @test_throws ErrorException lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation), Metida.CovarianceType(:XX)),
    )
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation), Metida.DIAG),
    )
    @test_throws ErrorException Metida.fit!(lmm; init = [0.0, 1.0, 0.0, 0.0, 0.0])
end


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


include("ar.jl")
include("lme4.jl")
include("norand.jl")
