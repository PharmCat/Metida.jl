path    = dirname(@__FILE__)

remls = [530.1445193510292, -30.67455875307806, 425.4465604691695, 314.22176883261096, -74.87997833266812, 530.1445193182162, 1387.0928273412144, 2342.5993980030553, 2983.26033032097, -16.41729812792036,
250.94514897106058, 1140.3816624784859, 2087.481017283834 , 1012.351698923092, 2087.481017283834 , 323.99766668546584, 77.56902301272578, 904.8743799636109 , 782.9395904949903 ,796.3124436472704,
470.59083255259935,248.99027587947566,119.80621157945501,274.30636135925795,660.0465401679254,433.84147581860896,1123.6556434756412,329.2574937705332,26.96606070210349,26.316526650535426]

remlsb = [536.2011487949176,-28.588171019268458,436.5116765557768,316.05056715993294,-74.54174147034092,536.2011487949178,1388.5367906718716,2345.962740738334,2985.724169788945,-15.89382595425026,
253.0510786936172,1143.3630490411597, 2090.103960238728,1050.3094549379975,2090.103960238728,326.53614303770223,78.84472797596732,937.9855867469028,815.2250801964899,827.9845298841103,
474.08826545614767,249.2537119063981,129.16329544316406,283.59476106417014,678.7453061894956,435.44528875067806,1125.0834323641677,331.8084510486417,45.304914448494124,32.41804784450319]
#8, 9, 12
#15?
#! 16, 27

#=
SPSS
REML G - CSH, R - DIAG
13 - 2087.4810139727 / 2087.481017533285
15 - / 2087.481017533285 / 2087.4810192819373
16 - 323.99766668073 / 323.9976737508006
29 - / 26.96606070210349
30 - 14.944038069158

REML G - SI, R - SI
4 - 316.050567
27 - 1125.092383
30 - 32.418048

=#
for i = 1:30
    dfrds        = CSV.File(path*"/csv/berds/rds"*string(i)*".csv", types = Dict(:PK => Float64, :subject => String, :period => String, :sequence => String, :treatment => String )) |> DataFrame
    dropmissing!(dfrds)

    dfrds.lnpk = log.(dfrds.PK)

    @testset "  RDS Test $(i)                                               " begin
        atol=1E-6

        lmm = Metida.LMM(@formula(lnpk~sequence+period+treatment), dfrds;
        random = Metida.VarEffect(Metida.@covstr(1|subject), Metida.SI),
        )
        Metida.fit!(lmm)
        @test lmm.result.reml ≈ remlsb[i] atol=atol

        if i ∈ [13, 15] atol = 1E-4 end
        lmm = Metida.LMM(@formula(lnpk~sequence+period+treatment), dfrds;
        random = Metida.VarEffect(Metida.@covstr(treatment|subject), Metida.CSH),
        repeated = Metida.VarEffect(Metida.@covstr(treatment|subject), Metida.DIAG),
        )
        Metida.fit!(lmm)
        @test lmm.result.reml ≈ remls[i] atol=atol
    end
end
