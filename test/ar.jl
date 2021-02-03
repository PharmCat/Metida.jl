
#=
SPSS
REML 453.339544
=#
df        = CSV.File(path*"/csv/RepeatedPulse.csv"; types = [String, Float64, String, String]) |> DataFrame
df.Pulse = float.(df.Pulse)
sort!(df, :Day)

@testset "  AR RepeatedPulse.csv                                     " begin
    lmm = Metida.LMM(@formula(Pulse~1), df;
    random = Metida.VarEffect(Metida.@covstr(Time), Metida.SI),
    repeated = Metida.VarEffect(Metida.@covstr(Day), Metida.AR),
    subject = :Time
    )
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 453.3395435627574 atol=1E-6

    lmm = Metida.LMM(@formula(Pulse~1), df;
    repeated = Metida.VarEffect(Metida.@covstr(1), Metida.AR),
    subject = :Time
    )
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 471.85107712169827 atol=1E-6

    lmm = Metida.LMM(@formula(Pulse~1), df;
    random = Metida.VarEffect(Metida.@covstr(Day), Metida.AR),
    subject = :Time
    )
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 453.3395560121246 atol=1E-6
end

#=
SPSS
REML 5456.638120
=#
df        = CSV.File(path*"/csv/ChickWeight.csv"; types = [String, Float64, Float64, String, String]) |> DataFrame

@testset "  ARH ChickWeight.csv                                      " begin
    lmm = Metida.LMM(@formula(weight~1 + Diet & Time), df;
    random = Metida.VarEffect(Metida.@covstr(1), Metida.SI),
    repeated = Metida.VarEffect(Metida.@covstr(Diet), Metida.ARH),
    subject = :Chick
    )
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 4439.254893054906 atol=1E-6

    lmm = Metida.LMM(@formula(weight~1 + Diet & Time), df;
    random = Metida.VarEffect(Metida.@covstr(Diet), Metida.ARH),
    subject = :Chick
    )
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 5480.751465914058 atol=1E-6
end
