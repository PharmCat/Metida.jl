# Metida

using  Test, CSV, DataFrames, StatsModels

path    = dirname(@__FILE__)
include("testdata.jl")

@testset "  Model: VC + nothing                                      " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation), Metida.VC),
    )
    Metida.fit!(lmm)

    io = IOBuffer();
    Base.show(io, lmmr)
    @test true
end
@testset "  Model: VC/subject + nothing                              " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation), Metida.VC),
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
    lmmr = Metida.fit!(lmm)
    @test true
end
@testset "  Model: (VC(period) + VC(formulation)) + nothing          " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = [Metida.VarEffect(Metida.@covstr(period), Metida.VC), Metida.VarEffect(Metida.@covstr(formulation), Metida.VC)],
    )
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 25.12948063433108 atol=1E-8

end
@testset "  Model: CSH + VC                                          " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation), Metida.CSH),
    repeated = Metida.VarEffect(Metida.@covstr(formulation), Metida.VC),
    )
    lmmr = Metida.fit!(lmm)
    @test true
end
@testset "  Model: CSH + CSH                                         " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation), Metida.CSH),
    repeated = Metida.VarEffect(Metida.@covstr(formulation), Metida.CSH),
    )
    lmmr = Metida.fit!(lmm)
    @test true
end
@testset "  Model: CSH/subject + nothing                             " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation), Metida.CSH),
    subject = :subject
    )
    lmmr = Metida.fit!(lmm)
    @test true
end
@testset "  Model: CSH/subject + VC                                  " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation), Metida.CSH),
    repeated = Metida.VarEffect(Metida.@covstr(formulation), Metida.VC),
    subject = :subject
    )
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 10.065238620469083 atol=1E-8

end
@testset "  Model: SI/subject + VC                                   " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation), Metida.SI),
    repeated = Metida.VarEffect(Metida.@covstr(formulation), Metida.VC),
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
    lmmr = Metida.fit!(lmm)
    @test true
end
@testset "  Model: (CSH+VC) + nothing                                " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = [Metida.VarEffect(Metida.@covstr(formulation), Metida.CSH),
    Metida.VarEffect(Metida.@covstr(period), Metida.VC)],
    )
    lmmr = Metida.fit!(lmm)
    @test true
end
@testset "  Model: CSH(formulation + period) + nothing               " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation + period), Metida.CSH),
    )
    lmmr = Metida.fit!(lmm)
    @test true
end
@testset "  Model: CSH(formulation & period) + nothing               " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation & period), Metida.CSH),
    )
    #lmmr = Metida.fit!(lmm)
    @test true
end
@testset "  Model: CSH(formulation * period) + nothing               " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation * period), Metida.CSH),
    )
    #lmmr = Metida.fit!(lmm)
    @test true
end
