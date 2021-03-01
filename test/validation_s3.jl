#=
remls = [530.1445193510292, -30.67455875307806, 425.4465604691695, 314.22176883261096, -74.87997833266812, 530.1445193182162, 1387.0928273412144, 2342.5993980030553, 2983.26033032097, -16.41729812792036,
250.94514897106058, 1140.3816624784859, 2087.481017283834 , 1012.351698923092, 2087.481017283834 , 323.99766668546584, 77.56902301272578, 904.8743799636109 , 782.9395904949903 ,796.3124436472704,
470.59083255259935,248.99027587947566,119.80621157945501,274.30636135925795,660.0465401679254,433.84147581860896,1123.6556434756412,329.2574937705332,26.96606070210349,26.316526650535426]

remlsb = [536.2011487949176,-28.588171019268458,436.5116765557768,316.05056715993294,-74.54174147034092,536.2011487949178,1388.5367906718716,2345.962740738334,2985.724169788945,-15.89382595425026,
253.0510786936172,1143.3630490411597, 2090.103960238728,1050.3094549379975,2090.103960238728,326.53614303770223,78.84472797596732,937.9855867469028,815.2250801964899,827.9845298841103,
474.08826545614767,249.2537119063981,129.16329544316406,283.59476106417014,678.7453061894956,435.44528875067806,1125.0834323641677,331.8084510486417,45.304914448494124,32.41804784450319]
=#
remlsb = zeros(Float64, 30)

remlsc = zeros(Float64, 30)

remlsrb = [536.20114880,-28.58817102,436.51167656,316.05056716,-74.54174147,
           536.20114880,1388.53679067,2345.96274074,2985.72416979,-15.89382595,
           253.05107869,1143.36304904,2090.10396024,1050.30945494,2090.10396024,
           326.53614304,78.84472798,937.98558675,815.22508020,827.98452988,
           474.08826546,249.25371191,129.16329544,283.59476106,678.74530619,
           435.44528875,1125.09238307,331.80845105,45.30491445,32.41804784]

remlsrc= [530.14451859,-30.67456491,425.44656047,314.22176883,-74.87997833,
          530.14451859,1387.09282701,2342.59939800,2983.26033032,-16.41729813,
           250.94514897,1140.38166248,2087.48101397,1012.35169892,2087.48101397,
           323.99766668,77.56902301,904.87437996,782.93959049,796.31244365,
           470.59083255,248.99027588,119.80621158,274.30636136,660.04654017,
           433.84147582,1123.66435096,329.25749377,26.96606070,26.3165]

commentsc1 = [1,3,6,13,15,16,24,25]
commentsc2 = [2,5,7,10,22,30]

c1 =
"The final Hessian matrix is not positive definite although all convergence criteria
are satisfied. The MIXED procedure continues despite this warning. Validity of subsequent
results cannot be ascertained."
c2 =
"Iteration was terminated but convergence has not been achieved. The MIXED procedure
continues despite this warning. Subsequent results produced are based on the last iteration.
Validity of the model fit is uncertain."
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
@testset "  RDS Test                                                 " begin
for i = 1:30
    dfrds        = CSV.File(joinpath(path, "csv", "berds", "rds"*string(i)*".csv"), types = Dict(:PK => Float64, :subject => String, :period => String, :sequence => String, :treatment => String )) |> DataFrame
    dropmissing!(dfrds)
    dfrds.lnpk = log.(dfrds.PK)
    @testset "  RDS Test $(i)                                               " begin
        atol=1E-6
        if i ∈ [27] atol = 1E-2 end
        lmm = Metida.LMM(@formula(lnpk~sequence+period+treatment), dfrds;
        random = Metida.VarEffect(Metida.@covstr(1|subject), Metida.SI),
        )
        Metida.fit!(lmm)
        @test lmm.result.reml ≈ remlsrb[i] atol=atol
        remlsb[i] = Metida.m2logreml(lmm)

        atol=1E-6
        if i ∈ [2, 13, 15] atol = 1E-4 end
        if i ∈ [27, 30] atol = 1E-2 end
        lmm = Metida.LMM(@formula(lnpk~sequence+period+treatment), dfrds;
        random = Metida.VarEffect(Metida.@covstr(treatment|subject), Metida.CSH),
        repeated = Metida.VarEffect(Metida.@covstr(treatment|subject), Metida.DIAG),
        )
        Metida.fit!(lmm)
        @test lmm.result.reml ≈ remlsrc[i] atol=atol
        remlsc[i] = Metida.m2logreml(lmm)
    end
end
end
remlsrc[30] = 14.94403807
dftable = DataFrame(n = collect(1:30), a = remlsb, b = remlsrb, c = Vector{Any}(undef, 30), d = remlsc, e = remlsrc, f = Vector{Any}(undef, 30), g = fill!(Vector{String}(undef, 30),""))

for i = 1:30
    if isapprox(dftable.a[i], dftable.b[i]; atol=1E-6)  dftable.c[i] = "OK" else dftable.c[i] = dftable.a[i] - dftable.b[i] end
    if isapprox(dftable.d[i], dftable.e[i]; atol=1E-6)  dftable.f[i] = "OK" else dftable.f[i] = dftable.d[i] - dftable.e[i] end
    if i in commentsc1
        dftable.g[i] = "*"
    end
    if i in commentsc2
        dftable.g[i] = "**"
    end
end

################################################################################


dfrds        = CSV.File(joinpath(path, "csv", "berds", "rds"*string(27)*".csv"), types = Dict(:PK => Float64, :subject => String, :period => String, :sequence => String, :treatment => String )) |> DataFrame
dropmissing!(dfrds)
dfrds.lnpk = log.(dfrds.PK)
fm2 = @formula(lnpk~sequence+period+treatment+(1|subject))
mm  = fit(MixedModel, fm2, dfrds, REML=true)

println("")
pretty_table(dftable, ["RDS" "REML B" "REML B" "DIFF B" "REML C" "REML C" "DIFF C" "Comm.";
                       " N " "Metida" " SPSS " "      " "Metida" " SPSS " "      " "     "])
println("")
println("*  - ", c1)
println("")
println("** - ", c2)
println("")
println("DataSet 27 - MixedModels.jl result:")
println("")
println(mm)
