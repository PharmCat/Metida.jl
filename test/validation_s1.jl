path    = dirname(@__FILE__)

################################################################################
#                             sleepstudy.csv
################################################################################

df        = CSV.File(path*"/csv/sleepstudy.csv"; types = [String, Float64, String, String]) |> DataFrame

@testset "  Model 1: sleepstudy.csv SI/SI                            " begin
    #=
    SPSS
    REML 1729.492560
    Random 1375.465318
    Residual 987.588488
    =#
    # Model 1
    lmm = Metida.LMM(@formula(Reaction~Days), df;
    random = Metida.VarEffect(Metida.@covstr(1|Subject), Metida.SI),
    )
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 1729.4925602367025 atol=1E-6
end

@testset "  Model 2: sleepstudy.csv CS/SI                            " begin
    # REML SPSS 1904.327
    # 1662.172084
    # 296.693108
    # Model 2
    lmm = Metida.LMM(@formula(Reaction~1), df;
    random = Metida.VarEffect(Metida.@covstr(Days|Subject), Metida.CS),
    )
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 1904.3265170722132 atol=1E-6

    #=
    lmm = Metida.LMM(@formula(Reaction~1 + Days), df;
    repeated = Metida.VarEffect(Metida.@covstr(1), Metida.CS, subj = :Subject)
    )
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 1729.4925602367027 atol=1E-6
    =#
end

@testset "  Model 3: sleepstudy.csv CSH/SI                           " begin
    # REML 1772.095
    # Model 3
    lmm = Metida.LMM(@formula(Reaction~1), df;
    random = Metida.VarEffect(Metida.@covstr(Days|Subject), Metida.CSH)
    )
    Metida.fit!(lmm; init = [26.4881, 35.5197, 34.8287, 56.1999, 63.8281, 85.4346, 94.218, 92.8584, 113.679, 129.721, 0.959643, 22.5597])
    @test lmm.result.reml ≈ 1772.0953251997046 atol=1E-6
end

@testset "  Model 4: sleepstudy.csv ARH/SI                           " begin
    # REML 1730.189543
    # Model 4
    lmm = Metida.LMM(@formula(Reaction~1), df;
    random = Metida.VarEffect(Metida.@covstr(Days|Subject), Metida.ARH)
    )
    Metida.fit!(lmm; init = [37.9896, 41.1392, 34.1041, 48.1435, 52.2191, 72.4237, 83.3405, 76.7782, 90.2571, 102.617, 0.900038, 6.83327])
    @test lmm.result.reml ≈ 1730.1895427398322 atol=1E-6
end

################################################################################
#                             Pastes.csv
################################################################################

df        = CSV.File(path*"/csv/Pastes.csv"; types = [String, Float64, String, String, String]) |> DataFrame
@testset "  Model 5: Pastes.csv SI,SI/SI                             " begin
    # REML 246.990746
    # Model 5
    lmm = Metida.LMM(@formula(strength~1), df;
    random = [Metida.VarEffect(Metida.@covstr(1|batch), Metida.SI),
    Metida.VarEffect(Metida.@covstr(1|batch&cask), Metida.SI)]
    )
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 246.99074585348623 atol=1E-6
end

@testset "  Model 6: Pastes.csv ARMA/SI                              " begin
    # SPSS REML 246.818951
    # Model 6
    lmm = Metida.LMM(@formula(strength~1), df;
    random = Metida.VarEffect(Metida.@covstr(cask|batch), Metida.ARMA),
    )
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 246.81895071012508 atol=1E-6
end

################################################################################
#                             Penicillin.csv
################################################################################

df        = CSV.File(path*"/csv/Penicillin.csv"; types = [String, Float64, String, String]) |> DataFrame
@testset "  Model 7: Penicillin.csv SI,SI/SI                         " begin
    # SPSS 330.860589
    # Model 7
    lmm = Metida.LMM(@formula(diameter ~ 1), df;
    random = [Metida.VarEffect(Metida.@covstr(1|plate), Metida.SI),
    Metida.VarEffect(Metida.@covstr(1|sample), Metida.SI)],
    )
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 330.86058899109184 atol=1E-6

    #=
    #SPSS 432.686
    lmm = Metida.LMM(@formula(diameter~1), df;
    random = Metida.VarEffect(Metida.@covstr(plate|sample), Metida.ARMA)
    )
    Metida.fit!(lmm)
    =#
end

################################################################################
#                             RepeatedPulse.csv
################################################################################

df        = CSV.File(path*"/csv/RepeatedPulse.csv"; types = [String, Float64, String, String]) |> DataFrame
df.Pulse = float.(df.Pulse)
sort!(df, :Day)

@testset "  Model 8: RepeatedPulse.csv SI/AR                         " begin
    #=
    SPSS
    REML 453.339544
    =#
    lmm = Metida.LMM(@formula(Pulse~1), df;
    random = Metida.VarEffect(Metida.@covstr(Time|Time), Metida.SI),
    repeated = Metida.VarEffect(Metida.@covstr(Day|Time), Metida.AR),
    )
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 453.3395435627574 atol=1E-6
end
@testset "  Model 9: RepeatedPulse.csv 0/AR                          " begin
    #=
    SPSS
    REML 471.851077
    =#
    lmm = Metida.LMM(@formula(Pulse~1), df;
    repeated = Metida.VarEffect(Metida.@covstr(Day|Time), Metida.AR),
    )
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 471.85107712169827 atol=1E-6
end
@testset "  Model 10: RepeatedPulse.csv AR/SI                        " begin
    #=
    SPSS
    REML 453.339555
    =#
    lmm = Metida.LMM(@formula(Pulse~1), df;
    random = Metida.VarEffect(Metida.@covstr(Day|Time), Metida.AR),
    )
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 453.3395560121246 atol=1E-6
end

################################################################################
#                             ChickWeight.csv
################################################################################
#=
SPSS
REML 5456.638120
=#
df        = CSV.File(path*"/csv/ChickWeight.csv"; types = [String, Float64, Float64, String, String]) |> DataFrame
sort!(df, :Diet)
@testset "  ARH ChickWeight.csv                                      " begin
    lmm = Metida.LMM(@formula(weight~1 + Diet & Time), df;
    random = Metida.VarEffect(Metida.@covstr(1|Chick), Metida.SI),
    repeated = Metida.VarEffect(Metida.@covstr(Diet|Chick), Metida.ARH),
    )
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 4439.254893054906 atol=1E-6

    lmm = Metida.LMM(@formula(weight~1 + Diet & Time), df;
    random = Metida.VarEffect(Metida.@covstr(Diet|Chick), Metida.ARH),
    )
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 5480.751465914058 atol=1E-6
end
