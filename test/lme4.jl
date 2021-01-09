

df        = CSV.File(path*"/csv/lme4/sleepstudy.csv") |> DataFrame
transform!(df, :Subject => categorical, renamecols=false)
transform!(df, :Days => categorical, renamecols=false)


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
transform!(df, :plate => categorical, renamecols=false)
transform!(df, :sample => categorical, renamecols=false)


@testset " Penicillin.csv" begin
    lmm = Metida.LMM(@formula(diameter~1), df;
    random = [Metida.VarEffect(Metida.SI, subj = :plate), Metida.VarEffect(Metida.SI, subj = :sample)]
    )
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 330.86058899109184 atol=1E-6
end


df        = CSV.File(path*"/csv/lme4/Pastes.csv") |> DataFrame
transform!(df, :batch => categorical, renamecols=false)
transform!(df, :sample => categorical, renamecols=false)
transform!(df, :cask=> categorical, renamecols=false)

@testset " Pastes.csv" begin
    lmm = Metida.LMM(@formula(strength~1), df;
    random = [Metida.VarEffect(Metida.SI, subj = :batch), Metida.VarEffect(Metida.SI, subj = [:batch,  :cask])]
    )
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 246.99074585348623 atol=1E-6
end
