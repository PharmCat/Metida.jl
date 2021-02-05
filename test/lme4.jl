################################################################################
#                             sleepstudy.csv
################################################################################

df        = CSV.File(path*"/csv/lme4/sleepstudy.csv"; types = [String, Float64, String, String]) |> DataFrame

@testset "  sleepstudy.csv                                           " begin
    #=
    SPSS
    REML 1729.492560
    R 1375.465318
    Res 987.588488
    =#
    #Model 1
    lmm = Metida.LMM(@formula(Reaction~Days), df;
    random = Metida.VarEffect(Metida.SI),
    subject = :Subject
    )
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 1729.4925602367025 atol=1E-6
end

@testset "  CS sleepstudy.csv                                        " begin
    #REML SPSS 1904.327
    # 1662.172084
    # 296.693108
    #Model 2
    lmm = Metida.LMM(@formula(Reaction~1), df;
    random = Metida.VarEffect(Metida.@covstr(Days), Metida.CS),
    subject = :Subject
    )
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 1904.3265170722132 atol=1E-6

    lmm = Metida.LMM(@formula(Reaction~1 + Days), df;
    repeated = Metida.VarEffect(Metida.@covstr(1), Metida.CS, subj = :Subject)
    )
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 1729.4925602367027 atol=1E-6

end

@testset "  CSH sleepstudy.csv                                       " begin
    #REML 1772.095
    #Model 3
    lmm = Metida.LMM(@formula(Reaction~1), df;
    random = Metida.VarEffect(Metida.@covstr(Days), Metida.CSH, subj = :Subject)
    )
    Metida.fit!(lmm; init = [26.4881, 35.5197, 34.8287, 56.1999, 63.8281, 85.4346, 94.218, 92.8584, 113.679, 129.721, 0.959643, 22.5597])
    @test lmm.result.reml ≈ 1772.0953251997046 atol=1E-6
end

@testset "  ARH sleepstudy.csv                                       " begin
    #REML 1730.189543
    #Model 4
    lmm = Metida.LMM(@formula(Reaction~1), df;
    random = Metida.VarEffect(Metida.@covstr(Days), Metida.ARH, subj = :Subject)
    )
    Metida.fit!(lmm; init = [37.9896, 41.1392, 34.1041, 48.1435, 52.2191, 72.4237, 83.3405, 76.7782, 90.2571, 102.617, 0.900038, 6.83327])
    @test lmm.result.reml ≈ 1730.1895427398322 atol=1E-6
end


################################################################################
#                             Penicillin.csv
################################################################################

df        = CSV.File(path*"/csv/lme4/Penicillin.csv"; types = [String, Float64, String, String]) |> DataFrame
df.diameter = float.(df.diameter)

@testset " SI + SI Penicillin.csv                                    " begin
    #SPSS 330.860589
    lmm = Metida.LMM(@formula(diameter ~ 1), df;
    random = [Metida.VarEffect(Metida.SI, subj = :plate), Metida.VarEffect(Metida.SI, subj = :sample)]
    )
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 330.86058899109184 atol=1E-6

    #SPSS 432.686
    #=
    lmm = Metida.LMM(@formula(diameter~1), df;
    random = Metida.VarEffect(Metida.@covstr(plate), Metida.ARMA, subj = :sample)
    )
    Metida.fit!(lmm)
    =#
end

################################################################################
#                             Pastes.csv
################################################################################

df        = CSV.File(path*"/csv/lme4/Pastes.csv"; types = [String, Float64, String, String, String]) |> DataFrame
@testset " SI + SI Pastes.csv                                        " begin
    #REML 246.990746
    lmm = Metida.LMM(@formula(strength~1), df;
    random = [Metida.VarEffect(Metida.SI, subj = :batch), Metida.VarEffect(Metida.SI, subj = [:batch,  :cask])]
    )
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 246.99074585348623 atol=1E-6

    #SPSS REML 246.818951
    lmm = Metida.LMM(@formula(strength~1), df;
    random = Metida.VarEffect(Metida.@covstr(cask), Metida.ARMA, subj = :batch),
    )
    Metida.fit!(lmm)
    @test lmm.result.reml ≈ 246.81895071012508 atol=1E-6
end
