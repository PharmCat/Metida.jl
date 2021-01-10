# Metida

using  Test, CSV, DataFrames, StatsModels

path    = dirname(@__FILE__)
include("testdata.jl")

@testset "  Basic test, DIAG / SI                                    " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation), Metida.DIAG),
    )
    Metida.fit!(lmm)
    io = IOBuffer();
    Base.show(io, lmm)
    Base.show(io, lmm.data)
    Base.show(io, lmm.result)
    Base.show(io, lmm.covstr)
    @test Metida.logreml(lmm)   ≈ -12.564740317165533 atol=1E-6
    @test Metida.m2logreml(lmm) ≈ 25.129480634331067 atol=1E-6
    @test lmm.result.reml       ≈ 25.129480634331063 atol=1E-6 #need check
end
@testset "  Model: DIAG/subject + nothing                            " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation), Metida.DIAG),
    subject = :subject
    )
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 16.241112644506067 atol=1E-8

end
@testset "  Model: SI + nothing                                      " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(subject), Metida.SI),
    )
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 10.862124583312674 atol=1E-8
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
@testset "  Model: CSH + DIAG                                        " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation), Metida.CSH),
    repeated = Metida.VarEffect(Metida.@covstr(formulation), Metida.DIAG),
    )
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 25.00077786912235 atol=1E-8
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
@testset "  Model: CSH/subject + DIAG                                " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation), Metida.CSH),
    repeated = Metida.VarEffect(Metida.@covstr(formulation), Metida.DIAG),
    subject = :subject
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm) ≈ 10.065239006121315 atol=1E-6
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

include("ar.jl")
include("lme4.jl")
include("norand.jl")
include("berds.jl")
