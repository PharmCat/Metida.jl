
#=
SPSS
REML 453.339544
=#
df        = CSV.File(path*"/csv/RepeatedPulse.csv") |> DataFrame
df.Pulse = float.(df.Pulse)
transform!(df, :Time => categorical, renamecols=false)
transform!(df, :Day => categorical, renamecols=false)

@testset "  RepeatedPulse.csv" begin
    lmm = Metida.LMM(@formula(Pulse~1), df;
    random = Metida.VarEffect(Metida.@covstr(Time), Metida.SI),
    repeated = Metida.VarEffect(Metida.@covstr(Day), Metida.AR),
    subject = :Time
    )
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 453.1926914954099 atol=1E-6

    lmm = Metida.LMM(@formula(Pulse~1), df;
    repeated = Metida.VarEffect(Metida.@covstr(1), Metida.AR),
    subject = :Time
    )
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 470.5275730199629 atol=1E-6
end

#=
SPSS
REML 5456.638120
=#
df        = CSV.File(path*"/csv/ChickWeight.csv") |> DataFrame
df.weight = float.(df.weight)
df.Time = float.(df.Time)
df.Time2 = copy(df.Time)
transform!(df, :Time2 => categorical, renamecols=false)
transform!(df, :Chick => categorical, renamecols=false)
transform!(df, :Diet => categorical, renamecols=false)
@testset "  ChickWeight.csv" begin
    lmm = Metida.LMM(@formula(weight~1 + Diet & Time), df;
    random = Metida.VarEffect(Metida.@covstr(1), Metida.SI),
    repeated = Metida.VarEffect(Metida.@covstr(Diet), Metida.ARH),
    subject = :Chick
    )
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 5451.857613990478 atol=1E-6

    lmm = Metida.LMM(@formula(weight~1 + Diet & Time), df;
    random = Metida.VarEffect(Metida.@covstr(Diet), Metida.ARH),
    subject = :Chick
    )
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 5480.751465914058 atol=1E-6
end
