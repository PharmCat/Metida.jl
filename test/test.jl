# Metida

using  Test, CSV, DataFrames, StatsModels

path    = dirname(@__FILE__)
include("testdata.jl")

@testset "  Model: VC + nothing                                      " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation), Metida.VC),
    )
    lmmr = Metida.fit!(lmm)
    @test true
end
@testset "  Model: SI + nothing                                      " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation), Metida.SI),
    )
    lmmr = Metida.fit!(lmm)
    @test true
end
@testset "  Model: CSH + nothing                                     " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation), Metida.CSH),
    )
    lmmr = Metida.fit!(lmm)
    @test true
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
    lmmr = Metida.fit!(lmm)
    @test true
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
