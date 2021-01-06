

df        = CSV.File(path*"/csv/RepeatedPulse.csv") |> DataFrame
df.Pulse = float.(df.Pulse)
categorical!(df, :Time);
categorical!(df, :Day);
@testset "  RepeatedPulse.csv" begin
    lmm = Metida.LMM(@formula(Pulse~1), df;
    random = Metida.VarEffect(Metida.@covstr(Time), Metida.SI),
    repeated = Metida.VarEffect(Metida.@covstr(1), Metida.AR),
    subject = :Time
    )
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 450.75260020816347 atol=1E-6
end


df        = CSV.File(path*"/csv/ChickWeight.csv") |> DataFrame
df.weight = float.(df.weight)
df.Time = float.(df.Time)
df.Time2 = copy(df.Time)
categorical!(df, :Time2);
categorical!(df, :Chick);
categorical!(df, :Diet);
@testset "  ChickWeight.csv" begin
    lmm = Metida.LMM(@formula(weight~1 + Diet & Time), df;
    random = Metida.VarEffect(Metida.@covstr(1), Metida.SI),
    repeated = Metida.VarEffect(Metida.@covstr(Diet), Metida.ARH),
    subject = :Chick
    )
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 5453.137479692881 atol=1E-6
end
