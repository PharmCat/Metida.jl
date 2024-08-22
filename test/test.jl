# Metida

using  Test, CSV, DataFrames, StatsModels, StatsBase, LinearAlgebra, CategoricalArrays, Random, StableRNGs

path    = dirname(@__FILE__)
include("testdata.jl")

@testset "  Publick API basic tests                                  " begin
    io = IOBuffer();
    transform!(df0, :formulation => categorical, renamecols=false)
    # Basic, no block
    df0.nosubj = ones(size(df0, 1))
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation|nosubj), Metida.DIAG),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm, ) ≈ 25.129480634331067 atol=1E-6
    # Test -2 reml for provided theta
    @test Metida.m2logreml(lmm, Metida.theta(lmm)) ≈ 25.129480634331067 atol=1E-6

    # Casuistic case - random
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation|1), Metida.DIAG),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm) ≈ 25.129480634331067 atol=1E-6

    # Casuistic case - repeated
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    repeated = Metida.VarEffect(Metida.@covstr(formulation|1), Metida.DIAG),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm) ≈ 25.00077786912235 atol=1E-6

    # Missing
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0m;
    random = Metida.VarEffect(Metida.@covstr(formulation|subject), Metida.DIAG),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm) ≈ 16.636012616466203 atol=1E-6

    # milmm = Metida.MILMM(lmm, df0m)
    # Basic, Subject block
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation|subject), Metida.DIAG),
    )
    Metida.fit!(lmm; aifirst = true)
    @test Metida.m2logreml(lmm) ≈ 16.241112644506067 atol=1E-6

    lmm = Metida.fit(Metida.LMM, @formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation|subject), Metida.DIAG),
    )
    @test Metida.m2logreml(lmm) ≈ 16.241112644506067 atol=1E-6

    lmm = Metida.fit(Metida.LMM, Metida.@lmmformula(var~0+sequence+period+formulation,
    random = formulation|subject:Metida.DIAG), df0)
    @test Metida.fixedeffn(lmm) == 3
    t3table = Metida.typeiii(lmm)
    @test length(t3table.name) == 3

    lmm = Metida.fit(Metida.LMM, Metida.@lmmformula(var~sequence+period+formulation,
    random = formulation|subject:Metida.DIAG), df0)
    @test Metida.m2logreml(lmm) ≈ 16.241112644506067 atol=1E-6
    @test Metida.fixedeffn(lmm) == 4

    t3table = Metida.typeiii(lmm;  ddf = :contain) # NOT VALIDATED
    t3table = Metida.typeiii(lmm;  ddf = :residual)
    t3table = Metida.typeiii(lmm)
    @test length(t3table.name) == 4
    ############################################################################
    ############################################################################
    # API test
    ############################################################################
    l = [0 0 1 0 0 0; 0 0 0 1 0 0; 0 0 0 0 1 0]
    @test Metida.logreml(lmm)   ≈ -8.120556322253035 atol=1E-6
    @test Metida.theta(lmm) ≈ [0.4473222800422779, 0.3673667558902593, 0.1850675552332174]
    @test lmm.θ ≈ [0.4473222800422779, 0.3673667558902593, 0.1850675552332174]
    @test lmm.β ≈ coef(lmm)
    @test isfitted(lmm) == true
    @test islinear(lmm) == true
    @test bic(lmm)              ≈ 24.558878811225412 atol=1E-6
    @test aic(lmm)              ≈ 22.241112644506067 atol=1E-6
    @test aicc(lmm)             ≈ 24.241112644506067 atol=1E-6
    @test Metida.caic(lmm)      ≈ 27.558878811225412 atol=1E-6
    @test dof_residual(lmm) == 14

    @test Metida.dof_satter(lmm, 6)   ≈ 5.81896814947982 atol=1E-2
    @test Metida.dof_satter(lmm)[end] ≈ 5.81896814947982 atol=1E-2
    @test Metida.dof_satter(lmm, [0 0 0 0 0 1]) ≈ 5.81896814947982 atol=1E-2
    @test Metida.dof_satter(lmm, l) ≈ 7.575447546211385 atol=1E-2
    @test Metida.fvalue(lmm, l) ≈  0.202727915619993 atol=1E-2
    @test Metida.dof_satter(lmm, Metida.lcontrast(lmm,3)) ≈ 7.575447546211385 atol=1E-2
    @test nobs(lmm) == 20
    @test Metida.thetalength(lmm) == 3
    @test Metida.rankx(lmm) == 6
    @test sum(Metida.gmatrix(lmm, 1)) ≈ 0.3350555603325126 atol=1E-6
    @test sum(Metida.rmatrix(lmm, 1)) ≈ 0.13699999248885292 atol=1E-6
    @test sum(Metida.vmatrix(lmm, 1)) ≈ 1.4772222338189034 atol=1E-6
    @test dof(lmm) == 7
    @test vcov(lmm)[1,1]              ≈ 0.11203611149231425 atol=1E-6
    @test stderror(lmm)[1]            ≈ 0.33471795812641164 atol=1E-6
    @test length(modelmatrix(lmm)) == 120
    @test isa(response(lmm), Vector)
    @test sum(Metida.hessian(lmm))    ≈ 1118.160713481362 atol=1E-2
    @test Metida.nblocks(lmm) == 5
    @test coefnames(lmm) == ["(Intercept)", "sequence: 2", "period: 2", "period: 3", "period: 4", "formulation: 2"]
    @test Metida.gmatrixipd(lmm)
    @test Metida.confint(lmm)[end][1] ≈ -0.7630380758015894 atol=1E-4
    @test Metida.confint(lmm, 6)[1] ≈ -0.7630380758015894 atol=1E-4
    @test Metida.confint(lmm; ddf = :residual)[end][1] ≈ -0.6740837049617738 atol=1E-4
    @test Metida.responsename(lmm) == "var"
    @test Metida.nblocks(lmm) == 5
    @test Metida.msgnum(lmm.log) == 3

    Metida.confint(lmm; ddf = :contain)[end][1] #NOT VALIDATED
    @test size(crossmodelmatrix(lmm), 1) == 6
    @test t3table.pval[4]          ≈ 0.7852154468081014 atol=1E-6
    ct = Metida.contrast(lmm, [0 0 1 0 0 0; 0 0 0 1 0 0; 0 0 0 0 1 0])
    @test t3table.pval[3] ≈ ct.pval[1]
    est = Metida.estimate(lmm, [0,0,0,0,0,1]; level = 0.9)
    est = Metida.estimate(lmm; level = 0.9)

    @test_nowarn formula(lmm)
    
    #  
    onefelmm = Metida.LMM(@formula(var~1), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation|subject), Metida.DIAG),
    )
    @test coefnames(onefelmm) == "(Intercept)"
    @test_nowarn show(io, onefelmm)
    ############################################################################
    # AI like algo
    Metida.fit!(lmm; aifirst = true, init = Metida.theta(lmm))
    @test Metida.m2logreml(lmm) ≈ 16.241112644506067 atol=1E-6

    # Score
    Metida.fit!(lmm; aifirst = :score)
    @test Metida.m2logreml(lmm) ≈ 16.241112644506067 atol=1E-6

    # AI
    Metida.fit!(lmm; aifirst = :ai)
    @test Metida.m2logreml(lmm) ≈ 16.241112644506067 atol=1E-6

    # Set user coding
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(1 + formulation|subject), Metida.CSH; coding = Dict(:formulation => StatsModels.DummyCoding())),
    )

    # Test varlink/rholinkf
    Metida.fit!(lmm; rholinkf = :sqsigm)
    @test Metida.dof_satter(lmm, [0, 0, 0, 0, 0, 1]) ≈ 6.043195705464293 atol=1E-2
    @test Metida.m2logreml(lmm) ≈ 10.314822559210157 atol=1E-6

    Metida.fit!(lmm; rholinkf = :atan)
    @test Metida.m2logreml(lmm) ≈ 10.314837309793571 atol=1E-6

    Metida.fit!(lmm; rholinkf = :psigm)
    @test Metida.m2logreml(lmm) ≈ 10.86212458333098 atol=1E-6

    Metida.fit!(lmm; varlinkf = :sq)
    @test Metida.m2logreml(lmm) ≈ 10.314822479530243 atol=1E-6

    # Repeated effect only
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    repeated = Metida.VarEffect(Metida.@covstr(formulation|nosubj)),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm) ≈ 25.129480634331063 atol=1E-6

    # Function term name
    lmm = Metida.fit(Metida.LMM, Metida.@lmmformula(log(var)~sequence+period+formulation,
    random = formulation|subject:Metida.DIAG), df0);
    @test  Metida.responsename(lmm) == "log(var)"

    # BE like
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation|subject), Metida.CSH),
    repeated = Metida.VarEffect(Metida.@covstr(formulation|subject), Metida.DIAG),
    )
    Metida.fit!(lmm; aifirst = :score)
    @test Metida.m2logreml(lmm) ≈ 10.065238626765524 atol=1E-6

    # One thread
    Metida.fit!(lmm; maxthreads = 1)
    @test Metida.m2logreml(lmm) ≈ 10.065238626765524 atol=1E-6

    # incomplete
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df1;
    random = Metida.VarEffect(Metida.@covstr(formulation|subject), Metida.CSH),
    repeated = Metida.VarEffect(Metida.@covstr(formulation|subject), Metida.DIAG),
    )
    Metida.fit!(lmm; hes = false)
    @test Metida.m2logreml(lmm) ≈ 14.819463206995163 atol=1E-6
    @test Metida.dof_satter(lmm, 6)   ≈ 3.981102548214154 atol=1E-2

    lmm = Metida.LMM(@formula(var~period*formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation+sequence|nosubj), Metida.SI),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm, [0.222283, 0.444566]) ≈ Metida.m2logreml(lmm) atol=1E-6

    # EXPERIMENTAL
    @test Metida.dof_contain(lmm, 1) == 12
    @test Metida.dof_contain(lmm, 5) == 8
    tt = Metida.typeiii(lmm)
    @test tt.f[2] ≈ 0.185268  atol=1E-5
    @test tt.ndf[2] ≈ 3.0 atol=1E-5
    @test tt.df[2] ≈ 3.39086 atol=1E-5
    @test tt.pval[2] ≈ 0.900636 atol=1E-5

    # Int dependent variable, function Term in random part
    df0.varint = Int.(ceil.(df0.var2))
    lmmint =  @test_warn "Response variable not <: AbstractFloat" Metida.fit(Metida.LMM, Metida.@lmmformula(varint~formulation,
    random = 1+var^2|subject:Metida.SI), df0)
    Metida.fit!(lmmint)
    @test Metida.m2logreml(lmmint) ≈ 84.23373276096902 atol=1E-6

    # Wts
    df0.wtsc = fill(0.5, size(df0, 1))
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation|subject), Metida.DIAG),
    wts = df0.wtsc)
    fit!(lmm)
    @test Metida.m2logreml(lmm) ≈ 16.241112644506067 atol=1E-6
    
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation|subject), Metida.DIAG),
    wts = "wts")
    fit!(lmm)
    @test Metida.m2logreml(lmm) ≈ 17.823729 atol=1E-6 # TEST WITH SPSS 28

    @test_warn "wts count not equal observations count! wts not used." lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation|subject), Metida.DIAG),
    wts = ones(10))

    # Matrix wts
    matwts = Symmetric(rand(size(df0,1), size(df0,1)))
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation|subject), Metida.DIAG),
    wts = matwts)
    @test_nowarn fit!(lmm)

    # experimental weighted covariance 
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    repeated = Metida.VarEffect(Metida.@covstr(1|subject), Metida.SWC(matwts)))
    @test_nowarn fit!(lmm)
    @test_nowarn show(io, lmm)

    # Repeated vector
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    repeated = [Metida.VarEffect(Metida.@covstr(formulation|subject), Metida.DIAG), Metida.VarEffect(Metida.@covstr(1|subject), Metida.SI)])
    fit!(lmm)
    @test_nowarn show(io, lmm)

end
################################################################################
#                                  df0
################################################################################
@testset "  Model: Only repeated, 0/DIAG                             " begin
    # nowarn
    io = IOBuffer();
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    repeated = Metida.VarEffect(Metida.@covstr(formulation|subject), Metida.DIAG),
    )
    Metida.fit!(lmm)
    Base.show(io, lmm)
    @test Metida.m2logreml(lmm) ≈ 25.000777869122338 atol=1E-8
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    repeated = Metida.VarEffect(Metida.@covstr(formulation|subject), Metida.DIAG)
    )
    Metida.fit!(lmm)
    std  = stderror(lmm)
    cn   = coef(lmm)
    @test Metida.m2logreml(lmm) ≈ 25.000777869122338 atol=1E-8

    @test std[1] ≈ 0.2593212327384077 atol=1E-8
    @test cn[1]  ≈ 1.6213181171718132 atol=1E-8
end
@testset "  Model: Only repeated, noblock, 0/CSH (rholinkf = :atan)  " begin
    # nowarn
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    repeated = Metida.VarEffect(Metida.@covstr(period|subject), Metida.CSH),
    )
    Metida.fit!(lmm; rholinkf = :atan)
    std  = stderror(lmm)
    cn   = coef(lmm)
    @test Metida.m2logreml(lmm) ≈ 8.740095378772942 atol=1E-8

    @test std[1] ≈ 0.28779019255752775 atol=1E-8
    @test cn[1]  ≈ 1.3128476653830754 atol=1E-8

end
@testset "  Model: Only random, noblock, SI/SI                       " begin
    # nowarn
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(subject|nosubj), Metida.SI),
    )
    Metida.fit!(lmm)
    std  = stderror(lmm)
    cn   = coef(lmm)
    @test Metida.m2logreml(lmm) ≈ 10.862124583312674 atol=1E-8

    @test std[1] ≈ 0.30977407048924344 atol=1E-8
    @test cn[1]  ≈ 1.610000000000001 atol=1E-8
end
@testset "  Model: Only random, INT, SI/SI                           " begin
    # nowarn
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(1|subject), Metida.SI),
    )
    Metida.fit!(lmm)
    std  = stderror(lmm)
    cn   = coef(lmm)
    @test Metida.m2logreml(lmm) ≈ 10.862124583312674 atol=1E-8

    @test std[1] ≈ 0.3097740704892435 atol=1E-8
    @test cn[1]  ≈ 1.609999999999999 atol=1E-8
end
@testset "  Model: Noblock, equal subjects, CSH/CS + UN euqiv        " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation|subject), Metida.CSH),
    repeated = Metida.VarEffect(Metida.@covstr(formulation|subject), Metida.CS),
    )
    Metida.fit!(lmm)
    std  = stderror(lmm)
    cn   = coef(lmm)
    @test Metida.m2logreml(lmm) ≈ 10.3039977509049 atol=1E-6 #need check

    @test std[1] ≈ 0.33581840553609543 atol=1E-8
    @test cn[1]  ≈ 1.6100000000000012 atol=1E-8


    lmm_un = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation|subject), Metida.UN),
    repeated = Metida.VarEffect(Metida.@covstr(formulation|subject), Metida.CS),
    )
    Metida.fit!(lmm_un)
    @test Metida.m2logreml(lmm) ≈ Metida.m2logreml(lmm_un)
end
@testset "  Model: Different subjects, INT, CSH/DIAG                 " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(1 + formulation|subject), Metida.CSH; coding = Dict(:formulation => DummyCoding())),
    repeated = Metida.VarEffect(Metida.@covstr(formulation|subject&period), Metida.DIAG),
    )
    Metida.fit!(lmm)
    std  = stderror(lmm)
    cn   = coef(lmm)
    @test Metida.m2logreml(lmm) ≈ 10.06523870216023 atol=1E-4 #need check

    @test std[1] ≈ 0.3345433916523553 atol=1E-8
    @test cn[1]  ≈ 1.577492862311838 atol=1E-8
end
@testset "  Model: CSH/DIAG (rholinkf = :psigm) & lmmformula         " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation|subject), Metida.CSH),
    repeated = Metida.VarEffect(Metida.@covstr(formulation|subject), Metida.DIAG),
    )
    Metida.fit!(lmm; rholinkf = :psigm)
    std  = stderror(lmm)
    cn   = coef(lmm)
    @test Metida.m2logreml(lmm) ≈ 10.065239006121315 atol=1E-6

    @test std[1] ≈ 0.3345433910321999 atol=1E-8
    @test cn[1]  ≈ 1.5774928621922844 atol=1E-8
    std  = stderror(lmm)

    lmm = Metida.LMM(Metida.@lmmformula(var~sequence+period+formulation,
    random = formulation|subject:Metida.CSH,
    repeated = formulation|subject:Metida.DIAG),
    df0)
    Metida.fit!(lmm; rholinkf = :psigm)
    @test Metida.m2logreml(lmm) ≈ 10.065239006121315 atol=1E-6
end
################################################################################
#                                  ftdf / 1fptime.csv
################################################################################
@testset "  Model: Categorical * Continuous predictor, CSH/SI        " begin
    # nowarn
    lmm = Metida.LMM(@formula(response ~1 + factor*time), ftdf;
    random = Metida.VarEffect(Metida.@covstr(1 + time|subject&factor), Metida.CSH),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm) ≈ 1300.1807598168923 atol=1E-6
    @test coef(lmm) ≈ [22.13309710783416, 2.000486297455917, 1.1185284725578566, 0.4049714576872601] atol=1E-6
    @test Metida.dof_satter(lmm, [0, 0, 0, 1]) ≈ 37.999999999991786 atol=1E-2
    #Metida.typeiii(lmm)
end

@testset "  Model: Function terms, CSH/SI                            " begin
    ftdf.expresp = exp.(ftdf.response)
    ftdf.exptime = exp.(ftdf.time)
    lmm = Metida.LMM(@formula(log(expresp) ~ 1 + factor*log(exptime)), ftdf;
    random = Metida.VarEffect(Metida.@covstr(1 + log(exptime)|subject&factor), Metida.CSH),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm) ≈ 1300.1807598168923 atol=1E-6
end
################################################################################
#                                  ftdf2 / 1freparma.csv
################################################################################
@testset "  Model: Categorical * Continuous predictor, 0/ARMA        " begin
    # nowarn
    # SPSS 715.452856
    io = IOBuffer();
    lmm = Metida.LMM(@formula(response ~ 1 + factor*time), ftdf2;
    repeated = Metida.VarEffect(Metida.@covstr(time|subject&factor), Metida.ARMA),
    )
    Metida.fit!(lmm)
    println(io, lmm.log)
    @test Metida.m2logreml(lmm) ≈ 715.4528559688382 atol = 1E-6
end
@testset "  Model: Categorical * Continuous predictor, DIAG/AR       " begin
    # nowarn
    lmm = Metida.LMM(@formula(response ~ 1 + factor*time), ftdf2;
    random = Metida.VarEffect(Metida.@covstr(factor|subject&factor), Metida.DIAG),
    repeated = Metida.VarEffect(Metida.@covstr(1|subject&factor), Metida.AR),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm) ≈ 710.0962305879676 atol=1E-6
end
@testset "  Model: Categorical * Continuous predictor, 0/ARH         " begin
    # nowarn
    lmm = Metida.LMM(@formula(response ~ 1 + factor*time), ftdf2;
    repeated = Metida.VarEffect(Metida.@covstr(factor|subject&factor), Metida.ARH),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm) ≈ 731.7794071577566 atol=1E-6
end
################################################################################
#                                  ftdf3 / 2f2rand.csv
################################################################################
@testset "  Model: CS, CS/SI                                         " begin
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3; contrasts=Dict(:factor => DummyCoding(; base=1.0)),
    random = [Metida.VarEffect(Metida.@covstr(r1|subject), Metida.CS), Metida.VarEffect(Metida.@covstr(r2|subject), Metida.CS)],
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm)  ≈ 710.4250214813896 atol=1E-8
    # Ubuntu 1.8 x64 ≈ 20.881858029086246
    # Removed because unstable
    # @test Metida.dof_satter(lmm)[2] ≈ 20.94587351111687 atol=1E-8
    # Test multiple random effect γ
    @test_nowarn Metida.raneff(lmm, 1)
end
@testset "  Model: SI, SI/CSH                                        " begin
    # no errors
    # not validated
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3; contrasts=Dict(:factor => DummyCoding(; base=1.0)),
    random = [Metida.VarEffect(Metida.@covstr(1|subject), Metida.SI),
    Metida.VarEffect(Metida.@covstr(1|r1&subject), Metida.SI)],
    repeated = Metida.VarEffect(Metida.@covstr(p|subject), Metida.CSH)
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm)  ≈ 697.2241355154041 atol=1E-8

    lmmf = Metida.@lmmformula(response ~ 1 + factor,
    random = 1|subject/r1,
    repeated = p|subject:Metida.CSH)

    lmm = Metida.LMM(lmmf,
    ftdf3; contrasts=Dict(:factor => DummyCoding(; base=1.0)))

    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm)  ≈ 697.2241355154041 atol=1E-8
    io = IOBuffer();
    @test_nowarn show(io, lmmf)
    @test Metida.dof_satter(lmm)[2] ≈ 21.944891442712407 atol=1E-8
    # Test multiple random effect γ
    @test_nowarn Metida.raneff(lmm)
end
@testset "  Model: AR/SI                                             " begin
    # SPSS 698.879
    # nowarn
    #=
    MIXED response BY factor r1 subject p 
  /CRITERIA=DFMETHOD(SATTERTHWAITE) CIN(95) MXITER(100) MXSTEP(10) SCORING(1) 
    SINGULAR(0.000000000001) HCONVERGE(0.00000001, RELATIVE) LCONVERGE(0, ABSOLUTE) PCONVERGE(0, 
    ABSOLUTE) 
  /FIXED=factor | SSTYPE(3) 
  /METHOD=REML 
  /PRINT=SOLUTION 
  /RANDOM=r1 | SUBJECT(subject) COVTYPE(AR1) SOLUTION 
  /REPEATED=p | SUBJECT(subject) COVTYPE(DIAG).
    =#
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3; contrasts=Dict(:factor => DummyCoding(; base=1.0)),
    random = Metida.VarEffect(Metida.@covstr(r1|subject), Metida.AR),
    repeated = Metida.VarEffect(Metida.@covstr(p|subject), Metida.DIAG),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm)  ≈ 698.8792511057682 atol=1E-8
    #SPSS 22.313
    @test Metida.dof_satter(lmm)[2] ≈ 22.43888645153638 atol=1E-8
    #SPSS 
    re = Metida.raneff(lmm, 1)
    @test re[1][1][2][1] ≈ 2.147751 atol=1E-5
    @test re[1][1][2][2] ≈ 1.446182 atol=1E-5
    @test re[1][1][2][3] ≈ 1.496007 atol=1E-5
end

@testset "  Model: ARMA/SI                                           " begin
    io = IOBuffer();
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3; contrasts=Dict(:factor => DummyCoding(; base=1.0)),
    random = Metida.VarEffect(Metida.@covstr(p|r1&r2), Metida.ARMA),
    )
    Metida.fit!(lmm; verbose = 3, io = io)
    #[1.2964e-5, 0.0299594, 0.0699728, 3.69557]
    println(io, lmm.log)
    @test Metida.m2logreml(lmm)  ≈ 913.9176298311813 atol=1E-8
    #SPSS 166
    @test Metida.dof_satter(lmm)[2] ≈ 165.99999999999005 atol=1E-8
end

@testset "  Model: ARH/SI (subjects with &)                          " begin
    # SPSS 707.377
    # nowarn
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3; contrasts=Dict(:factor => DummyCoding(; base=1.0)),
    random = Metida.VarEffect(Metida.@covstr(r1|s2&factor), Metida.ARH),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm)  ≈ 707.3765873864152 atol=1E-8
    #SPSS 23.093
    @test Metida.dof_satter(lmm, [0, 1]) ≈ 23.111983305626193 atol=1E-2

    #SPSS 691.360073
    lmm = Metida.LMM(@formula(nrhoresp ~ 1 + factor), ftdf3; contrasts=Dict(:factor => DummyCoding(; base=1.0)),
    random = Metida.VarEffect(Metida.@covstr(r1|s2&factor), Metida.ARH),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm)  ≈ 691.3600726310308 atol=1E-8
    mtt = Metida.typeiii(lmm)
    #SPSS 48.550474
    @test mtt.df[2] ≈ 48.55470874755898 atol=1E-8

end
@testset "  Model: INT, *, DIAG/SI                                   " begin
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3;
    random = Metida.VarEffect(Metida.@covstr(1 + r2 * r1|subject), Metida.DIAG; coding=Dict(:r1 => DummyCoding(), :r2 => DummyCoding()))
    )
    Metida.fit!(lmm)
    @test Metida.theta(lmm)  ≈ [2.796694409004289, 2.900485570555582, 3.354913215348968, 2.0436114769223237, 1.8477830405766895, 2.0436115732330955, 1.0131934233937254] atol=1E-5 # atol=1E-8 !
    @test Metida.m2logreml(lmm)  ≈ 713.0655862252027 atol=1E-8
end
@testset "  Model: &, DIAG/SI                                        " begin
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3;
    random = Metida.VarEffect(Metida.@covstr(r1&r2|subject), Metida.DIAG),
    )
    Metida.fit!(lmm)
    @test Metida.theta(lmm)  ≈ [3.0325005960015985, 3.343826588448401, 1.8477830405766895, 1.8477830405766895, 1.8477830405766895, 4.462942536844632, 1.0082345219318216] atol=1E-5 # atol=1E-8 !
    @test Metida.m2logreml(lmm)  ≈ 719.9413776641368 atol=1E-8
end
@testset "  Model: INT, +,  TOEPHP(3)/SI                             " begin
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3;
    random = Metida.VarEffect(Metida.@covstr(1 + r1 + r2|subject), Metida.TOEPHP(3); coding = Dict(:r1 => DummyCoding(), :r2 => DummyCoding())),
    )
    Metida.fit!(lmm)
    @test Metida.theta(lmm)  ≈ [2.843269324925114, 3.3598654954863423, 7.582560427911907e-10, 4.133572859333964, -0.24881591201506625, 0.46067672264107506, 1.0091887333170306] atol=1E-8
    @test Metida.m2logreml(lmm)  ≈ 705.9946274598822 atol=1E-8
end
@testset "  Model: TOEP/SI                                           " begin
    # SPSS 710.200
    # nowarn
    io = IOBuffer();
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3;
    random = Metida.VarEffect(Metida.@covstr(r1|subject), Metida.TOEP),
    )
    Metida.fit!(lmm)
    Base.show(io, lmm)
    @test Metida.m2logreml(lmm)  ≈ 710.1998669150806 atol=1E-8
end
@testset "  Model: TOEPP(2)/SI                                       " begin
    # nowarn
    io = IOBuffer();
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3;
    random = Metida.VarEffect(Metida.@covstr(r1|subject), Metida.TOEPP(2)),
    )
    Metida.fit!(lmm)
    Base.show(io, lmm)
    @test Metida.m2logreml(lmm)  ≈ 715.2410264030134 atol=1E-8
end
@testset "  Model: DIAG/TOEPP(3)                                     " begin
    # nowarn
    io = IOBuffer();
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3;
    random = Metida.VarEffect(Metida.@covstr(r2|subject), Metida.DIAG),
    repeated = Metida.VarEffect(Metida.@covstr(p|subject), Metida.TOEPP(3)),
    )
    Metida.fit!(lmm)
    Base.show(io, lmm)
    @test Metida.m2logreml(lmm)  ≈ 773.9575538254085 atol=1E-8
end
@testset "  Model: TOEPH/SI                                          " begin
    # nowarn
    io = IOBuffer();
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3;
    random = Metida.VarEffect(Metida.@covstr(r1|subject), Metida.TOEPH),
    )
    Metida.fit!(lmm)
    Base.show(io, lmm)
    @test Metida.m2logreml(lmm)  ≈ 705.7916833009426 atol=1E-8
end
@testset "  Model: SI/TOEPHP(3)                                      " begin
    io = IOBuffer();
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3;
    random = Metida.VarEffect(Metida.@covstr(r1|subject), Metida.SI),
    repeated = Metida.VarEffect(Metida.@covstr(r1&r2|subject), Metida.TOEPHP(3)),
    )
    Metida.fit!(lmm)
    Metida.fit!(lmm; optmethod = Metida.LBFGS_OM)
    Base.show(io, lmm)
    @test Metida.m2logreml(lmm)  ≈ 713.5850978377632 atol=1E-8
end
@testset "  Model: BE RDS 1, FDA model                               "  begin
    dfrds        = CSV.File(joinpath(path, "csv", "berds", "rds1.csv"), types = Dict(:PK => Float64, :subject => String, :period => String, :sequence => String, :treatment => String )) |> DataFrame
    dropmissing!(dfrds)
    dfrds.lnpk = log.(dfrds.PK)
    lmm = Metida.LMM(@formula(lnpk~sequence+period+treatment), dfrds;
    random = Metida.VarEffect(Metida.@covstr(treatment|subject), Metida.CSH),
    repeated = Metida.VarEffect(Metida.@covstr(treatment|subject), Metida.DIAG),
    )
    Metida.fit!(lmm)
    @test collect(Metida.confint(lmm)[6]) ≈  [0.05379033790060175, 0.23713821749515449] atol=1E-8
    anovatable = Metida.typeiii(lmm)
    @test anovatable.pval ≈ [3.087934998046721e-63, 0.9176105002577626, 0.6522549061162943, 0.002010933915677479] atol=1E-4

    est = Metida.estimate(lmm, [0,0,0,0,0,1]; level = 0.9)
    @test est.t[1] ≈ 3.12818 atol=1E-4
    @test est.pval[1] ≈ 0.0020 atol=1E-4
    @test est.cil[1] ≈ 0.06863 atol=1E-4
    @test est.ciu[1] ≈ 0.2223 atol=1E-4

    lmm = Metida.LMM(@formula(lnpk~0+sequence+period+treatment), dfrds;
    random = Metida.VarEffect(Metida.@covstr(treatment|subject), Metida.CSH),
    repeated = Metida.VarEffect(Metida.@covstr(treatment|subject), Metida.DIAG),
    )
    Metida.fit!(lmm)
    anovatable = Metida.typeiii(lmm)
    @test anovatable.pval ≈ [0.9176105002855397, 0.6522549061174356, 0.0020109339157131302] atol=1E-4
end

@testset "  Model: BE RDS 1, 2X2 + UN test                           "  begin
    dfrds        = CSV.File(joinpath(path, "csv", "berds2x2", "rds1.csv"), types = Dict(:Var => Float64, :Subject => String, :Period => String, :Sequence => String, :Formulation => String )) |> DataFrame
    dropmissing!(dfrds)
    lmm = Metida.LMM(@formula(log(Var)~Sequence+Period+Formulation), dfrds;
    random = Metida.VarEffect(Metida.@covstr(1|Subject)),
    )
    Metida.fit!(lmm)
    anovatable = Metida.typeiii(lmm)
    @test Metida.m2logreml(lmm)  ≈ -1.0745407333692825 atol=1E-8

    # Unstructured
    lmm = Metida.LMM(@formula(log(Var)~Sequence+Period+Formulation), dfrds;
    repeated = Metida.VarEffect(Metida.@covstr(Formulation|Subject), Metida.UN),
    )
    Metida.fit!(lmm)
end


@testset "  Model: Custom covariance type                            " begin
    struct CustomCovarianceStructure <: Metida.AbstractCovarianceType end
    function Metida.covstrparam(ct::CustomCovarianceStructure, t::Int)::Tuple{Int, Int}
        return (t, 1)
    end
    function Metida.gmat!(mx, θ, ct::CustomCovarianceStructure)
        s = size(mx, 1)
        @inbounds @simd for m = 1:s
            mx[m, m] = θ[m]
        end
        if s > 1
            for m = 1:s - 1
                @inbounds @simd for n = m + 1:s
                    mx[m, n] = mx[m, m] * mx[n, n] * θ[end]
                end
            end
        end
        @inbounds @simd for m = 1:s
            mx[m, m] = mx[m, m] * mx[m, m]
        end
        nothing
    end
    # nowarn
    lmm = Metida.LMM(@formula(response ~1 + factor*time), ftdf;
    random = Metida.VarEffect(Metida.@covstr(1 + time|subject&factor), Metida.CovarianceType(CustomCovarianceStructure())),
    )
    Metida.fit!(lmm)
    reml_c = Metida.m2logreml(lmm)

    lmm = Metida.LMM(@formula(response ~1 + factor*time), ftdf;
    random = Metida.VarEffect(Metida.@covstr(1 + time|subject&factor), Metida.CSH),
    )
    Metida.fit!(lmm)
    reml = Metida.m2logreml(lmm)
    @test reml_c ≈ reml

    function Metida.rmat!(mx, θ, rz, ::CustomCovarianceStructure, ::Int)
        vec = Metida.tmul_unsafe(rz, θ)
        rn    = size(mx, 1)
        if rn > 1
            for m = 1:rn - 1
                @inbounds @simd for n = m + 1:rn
                    mx[m, n] += vec[m] * vec[n] * θ[end]
                end
            end
        end
            @inbounds  for m ∈ axes(mx, 1)
            mx[m, m] += vec[m] * vec[m]
        end
        nothing
    end

    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    repeated = Metida.VarEffect(Metida.@covstr(period|subject), CustomCovarianceStructure()),
    )
    Metida.fit!(lmm)
    io = IOBuffer();
    @test_nowarn show(io, lmm)
    @test Metida.m2logreml(lmm) ≈ 8.740095378772942 atol=1E-8
end

@testset "  Model: Spatial Exponential                               " begin
    lmm = Metida.LMM(@formula(response ~ 1), ftdf;
    repeated = Metida.VarEffect(Metida.@covstr(response+time|subject), Metida.SPEXP),
    )
    Metida.fit!(lmm)
    #SPSS 1528.715
    @test Metida.m2logreml(lmm) ≈ 1528.7150702624508 atol=1E-6
    @test Metida.dof_satter(lmm)[1] ≈ 17.719638497284286 atol=1E-2
    @test_nowarn Metida.fit!(lmm; varlinkf = :identity)
end

@testset "  Random                                                   " begin
    lmm = Metida.LMM(@formula(response ~ 1 + factor*time), ftdf2;
    random = Metida.VarEffect(Metida.@covstr(factor|subject&factor), Metida.DIAG),
    repeated = Metida.VarEffect(Metida.@covstr(1|subject&factor), Metida.AR),
    )
    Metida.fit!(lmm)
    #@test Metida.m2logreml(lmm) ≈ 710.0962305879676 atol=1E-6

    @test  mean(Metida.rand(StableRNG(1234), lmm)) ≈ 50.435413902238096
    Metida.rand(lmm)
    Metida.rand(lmm, [4.54797, 2.82342, 1.05771, 0.576979])
    Metida.rand(lmm, [4.54797, 2.82342, 1.05771, 0.576979], [44.3, 5.3, 0.5, 0.29])
    v = zeros(nobs(lmm))
    @test mean(Metida.rand!(StableRNG(1234), v, lmm)) ≈  50.435413902238096
    Metida.rand!(v, lmm)
    Metida.rand!(v, lmm, [4.54797, 2.82342, 1.05771, 0.576979])
    Metida.rand!(v, lmm, [4.54797, 2.82342, 1.05771, 0.576979], [44.3, 5.3, 0.5, 0.29])
end

@testset "  Show functions                                           " begin
    io = IOBuffer();
    @test_nowarn show(io, Metida.ScaledIdentity())
    @test_nowarn show(io, Metida.Diag())
    @test_nowarn show(io, Metida.Autoregressive())
    @test_nowarn show(io, Metida.HeterogeneousAutoregressive())
    @test_nowarn show(io, Metida.CompoundSymmetry())
    @test_nowarn show(io, Metida.HeterogeneousCompoundSymmetry())
    @test_nowarn show(io, Metida.AutoregressiveMovingAverage())
    @test_nowarn show(io, Metida.Toeplitz())
    @test_nowarn show(io, Metida.ToeplitzParameterized(3))
    @test_nowarn show(io, Metida.HeterogeneousToeplitz())
    @test_nowarn show(io, Metida.HeterogeneousToeplitzParameterized(3))
    @test_nowarn show(io, Metida.SpatialExponential())
    @test_nowarn show(io, Metida.SpatialPower())
    @test_nowarn show(io, Metida.SpatialGaussian())
    @test_nowarn show(io, Metida.Unstructured())
    @test_nowarn show(io, Metida.SpatialExponentialD())
    @test_nowarn show(io, Metida.SpatialPowerD())
    @test_nowarn show(io, Metida.SpatialGaussianD())
    @test_nowarn show(io, Metida.ZERO())

    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation|subject), Metida.CSH),
    repeated = Metida.VarEffect(Metida.@covstr(formulation|subject), Metida.DIAG),
    )
    Metida.fit!(lmm; rholinkf = :psigm, verbose = 2, io = io)

    @test_nowarn Base.show(io, lmm)
    @test_nowarn Base.show(io, lmm.data)
    @test_nowarn Base.show(io, lmm.result)
    @test_nowarn Base.show(io, lmm.covstr)
    @test_nowarn Base.show(io, lmm.covstr.repeated[1].covtype)
    @test_nowarn Base.show(io, Metida.getlog(lmm))

    t3table = Metida.typeiii(lmm)
    Base.show(io, t3table)

    est = Metida.estimate(lmm, [0,0,0,0,0,1]; level = 0.9)
    @test_nowarn Base.show(io, est)

    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    repeated = Metida.VarEffect(Metida.@covstr(1|subject), Metida.CSH),
    )
    Metida.fit!(lmm; rholinkf = :atan)
    @test Metida.m2logreml(lmm) ≈ 10.862124583312667 atol=1E-8
    @test_nowarn Base.show(io, lmm)
end
################################################################################
#                                  Errors
################################################################################
@testset "  Errors test                                              " begin
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation|nosubj), Metida.DIAG),
    )
    @test_throws ErrorException Metida.fit!(lmm; init = [1.0])
    @test_throws ErrorException Metida.hessian(lmm)
    @test_throws ErrorException Metida.dof_satter(lmm)
    @test_throws ErrorException Metida.confint(lmm)

    @test_throws ErrorException  Metida.LMM(@formula(var~sequence+period+formulation), df0;)

    @test_throws ErrorException  Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = [Metida.VarEffect(Metida.@covstr(formulation|nosubj), Metida.DIAG), Metida.VarEffect(Metida.@covstr(formulation|nosubj), Metida.RZero())]
    )

    @test_throws Metida.FormulaException lmm = Metida.LMM(@formula(response ~ 1 + factor*time), ftdf2;
    random = Metida.VarEffect(Metida.@covstr(factor|subject*factor), Metida.DIAG),
    repeated = Metida.VarEffect(Metida.@covstr(1|subject), Metida.ARMA),
    )
    @test_throws Metida.FormulaException lmm = Metida.LMM(@formula(response ~ 1 + factor*time), ftdf2;
    random = Metida.VarEffect(Metida.@covstr(factor|subject), Metida.DIAG),
    repeated = Metida.VarEffect(Metida.@covstr(1|subject+factor), Metida.ARMA),
    )

    @test_throws ErrorException  Metida.LMM(@formula(var~sequence+period+formulation), df0;)
    

    @test_throws ErrorException  begin
        # make cov type
        struct NewCCS <: Metida.AbstractCovarianceType end
        function Metida.covstrparam(ct::NewCCS, t::Int)::Tuple{Int, Int}
            return (t, 1)
        end
        # try to apply to repeated effect
        lmm = Metida.LMM(@formula(response ~1 + factor*time), ftdf;
        repeated = Metida.VarEffect(Metida.@covstr(1 + time|subject&factor), Metida.CovarianceType(NewCCS())),
        )
        # try to get V 
        Metida.vmatrix([1.0, 1.0, 1.0], lmm, 1) 
    end

    # Error messages
    io = IOBuffer();
    lmm = Metida.LMM(@formula(response ~ 1 + factor*time), ftdf2;
    random = Metida.VarEffect(Metida.@covstr(factor|subject&factor), Metida.DIAG),
    repeated = Metida.VarEffect(Metida.@covstr(1|subject&factor), Metida.ARMA),
    )
    Metida.fit!(lmm)
    println(io, lmm.log)
end
################################################################################
#                                  Sweep test
################################################################################
@testset "  Sweep operator test                                      " begin
    A =
[1.0  2  2  4  1
 2  2  3  3  5
 2  3  3  4  2
 4  3  4  4  5
 1  5  2  5  5]
    iA =  inv(A[1:4, 1:4])
    iAs = Symmetric(-Metida.sweep!(copy(A), 1:4)[1][1:4, 1:4])
    B = copy(A)
    for i = 1:4
        Metida.sweep!(B, i)
    end
    iAss = Symmetric(-B[1:4, 1:4])
    akk = zeros(5)
    iAb = Symmetric(-Metida.sweepb!(view(akk, 1:5), copy(A), 1:4)[1][1:4, 1:4])
    @test iA  ≈ iAs  atol=1E-6
    @test iA  ≈ iAss atol=1E-6
    @test iAs ≈ iAb  atol=1E-6
end

@testset "  Experimental                                             " begin

    io = IOBuffer();
    lmm = Metida.LMM(@formula(r2 ~ f), spatdf;
    repeated = Metida.VarEffect(Metida.@covstr(x+y|1), Metida.SPEXP),
    )
    Metida.fit!(lmm, maxthreads = 1)
    @test Metida.m2logreml(lmm) ≈ 1985.3417397854946 atol=1E-6
    @test Metida.dof_satter(lmm)[1] ≈ 10.261390893063432 atol=1E-2


    spatdf.ci = map(x -> CartesianIndex(x[:x], x[:y]), eachrow(spatdf))
    function Metida.edistance(mx::AbstractMatrix{<:CartesianIndex}, i::Int, j::Int)
        return sqrt((mx[i, 1][1] - mx[j, 1][1])^2 + (mx[i, 1][2] - mx[j, 1][2])^2)
    end
    lmm = Metida.LMM(@formula(r2 ~ f), spatdf;
    repeated = Metida.VarEffect(Metida.@covstr(ci|1), Metida.SPEXP; coding = Dict(:ci => Metida.RawCoding())),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm) ≈ 1985.3417397854946 atol=1E-6
    @test Metida.dof_satter(lmm)[1] ≈ 10.261390893063432 atol=1E-2


    lmm = Metida.LMM(@formula(r2 ~ f), spatdf;
    repeated = Metida.VarEffect(Metida.@covstr(x+y|1), Metida.SPPOW),
    )
    Metida.fit!(lmm, maxthreads = 1)
    @test Metida.m2logreml(lmm) ≈ 1985.3417397854946 atol=1E-6
    @test Metida.dof_satter(lmm)[1] ≈ 10.26139089306347 atol=1E-2
    #@test_nowarn Metida.fit!(lmm; varlinkf = :identity)

    lmm = Metida.LMM(@formula(r2 ~ f), spatdf;
    repeated = Metida.VarEffect(Metida.@covstr(x+y|1), Metida.SPGAU),
    )
    Metida.fit!(lmm, maxthreads = 1)
    show(io, lmm.log)
    @test Metida.m2logreml(lmm) ≈ 1924.1371609697842 atol=1E-6
    @test Metida.dof_satter(lmm)[1] ≈ 87.23260061576238 atol=1E-2

###############################################################################
    lmm = Metida.LMM(@formula(r4 ~ f), spatdf;
    repeated = Metida.VarEffect(Metida.@covstr(x+y|1), Metida.SPEXPD),
    )
    Metida.fit!(lmm, maxthreads = 1)
    @test Metida.m2logreml(lmm) ≈ 1835.8648295317691 atol=1E-6
    @test Metida.dof_satter(lmm)[1] ≈ 6.200022611925939 atol=1E-2

    lmm = Metida.LMM(@formula(r3 ~ f), spatdf;
    repeated = Metida.VarEffect(Metida.@covstr(x+y|1), Metida.SPPOWD),
    )
    Metida.fit!(lmm, maxthreads = 1)
    @test Metida.m2logreml(lmm) ≈ 1899.3636384223198 atol=1E-6
    @test Metida.dof_satter(lmm)[1] ≈ 58.794026556017556 atol=1E-2

    lmm = Metida.LMM(@formula(r5 ~ f), spatdf;
    repeated = Metida.VarEffect(Metida.@covstr(x+y|1), Metida.SPGAUD),
    )
    Metida.fit!(lmm, maxthreads = 1)
    @test Metida.m2logreml(lmm) ≈ 1860.4865219180099 atol=1E-6
    @test Metida.dof_satter(lmm)[1] ≈ 120.33321588847883 atol=1E-2
    Base.show(io, lmm)
    Base.show(io, lmm.log)
    Metida.raneff(lmm, 1)

    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0m;
    random = Metida.VarEffect(Metida.@covstr(formulation|subject), Metida.DIAG),
    )
    Metida.fit!(lmm)
    Metida.raneff(lmm, 1)

    #@test_nowarn Base.show(io, Metida.bootstrap(lmm; n = 10, double = false, verbose = false, rng = MersenneTwister(1263)))
    #@test_nowarn
    #Metida.METIDA_SETTINGS[:MAX_THREADS] = 1

    br = Metida.bootstrap(lmm; n = 10, double = false, verbose = false, rng = StableRNG(1234))
    br = Metida.bootstrap(lmm; n = 10, double = true, verbose = false, rng = StableRNG(1234))
    Base.show(io, br)
    confint(br)
    confint(br, 1; method = :bp)
    confint(br, 1; method = :rbp)
    confint(br, 1; method = :norm)
    confint(br, 1; method = :bcnorm)
    confint(br, 1; method = :jn)

    confint(br, 1; metric = :sd, method = :bp)
    confint(br, 1; metric = :theta, method = :bp)

    mi = Metida.MILMM(lmm, df0m)
    Base.show(io, mi)
    mir = Metida.milmm(mi; n = 10, verbose = false, rng = StableRNG(1234))
    Base.show(io, mir)

    @test_nowarn Metida.milmm(lmm, df0m; n = 10, verbose = false, rng = StableRNG(1234))

    @test_throws ErrorException Metida.milmm(lmm; n = 10, verbose = false, rng = StableRNG(1234))

    if !(VERSION < v"1.7")
        mb =  Metida.miboot(mi; n = 10, bootn = 10,  double = true, verbose = false, rng = StableRNG(1234))
        Base.show(io, mb)
    end

    # Other 
    @test Metida.varlinkvecapply([0.1, 0.1], [:var, :rho]; varlinkf = :exp, rholinkf = :sigm) ≈ [1.1051709180756477, 0.004999958333749888] atol=1E-6

end
