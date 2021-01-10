@testset "  No random                                                " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    repeated = Metida.VarEffect(Metida.@covstr(formulation), Metida.DIAG),
    subject = :subject
    )
    Metida.fit!(lmm)

    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    repeated = Metida.VarEffect(Metida.@covstr(formulation), Metida.DIAG, subj = :subject)
    )
    Metida.fit!(lmm)

    @test lmm.result.reml ≈ 25.000777869122338 atol=1E-8
end
