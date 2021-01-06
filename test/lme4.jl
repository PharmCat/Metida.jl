

df        = CSV.File(path*"/csv/lme4/sleepstudy.csv") |> DataFrame
categorical!(df, :Subject);
categorical!(df, :Days);

#=
SPSS
REML 1729.492560
=#
@testset "  sleepstudy.csv" begin
    lmm = Metida.LMM(@formula(Reaction~Days), df;
    random = Metida.VarEffect(Metida.SI),
    subject = :Subject
    )
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 1729.4925602367025 atol=1E-6
end

df        = CSV.File(path*"/csv/lme4/Penicillin.csv") |> DataFrame
df.diameter = float.(df.diameter)
categorical!(df, :plate);
categorical!(df, :sample);

@testset " Penicillin.csv" begin
    lmm = Metida.LMM(@formula(diameter~1), df;
    random = [Metida.VarEffect(Metida.SI, subj = :plate), Metida.VarEffect(Metida.SI, subj = :sample)]
    )
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 330.86058899109184 atol=1E-6
end


df        = CSV.File(path*"/csv/lme4/Pastes.csv") |> DataFrame
categorical!(df, :batch);
categorical!(df, :cask);
categorical!(df, :sample);
@testset " Pastes.csv" begin
    lmm = Metida.LMM(@formula(strength~1), df;
    random = [Metida.VarEffect(Metida.SI, subj = :batch), Metida.VarEffect(Metida.SI, subj = [:batch,  :cask])]
    )
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 246.99074585348623 atol=1E-6
end
