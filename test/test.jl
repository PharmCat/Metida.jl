# Metida

using  Test, CSV, DataFrames, StatsModels, StatsBase, LinearAlgebra, CategoricalArrays, Random, StableRNGs

path    = dirname(@__FILE__)
include("testdata.jl")

@testset "  Publick API basic tests                                  " begin

    io = IOBuffer()
    transform!(df0, :formulation => categorical, renamecols = false)
    df0.nosubj = ones(size(df0, 1))
    df0.varint = Int.(ceil.(df0.var2))
    df0.wtsc   = fill(0.5, size(df0, 1))
    matwts     = Symmetric(rand(StableRNG(20240501), size(df0, 1), size(df0, 1)))

    # Эталонная модель раздела: df0, случайный DIAG по формуляции внутри субъекта.
    refmodel() = Metida.LMM(@formula(var ~ sequence + period + formulation), df0;
        random = Metida.VarEffect(Metida.@covstr(formulation | subject), Metida.DIAG))

    REML0  = 16.241112644506067
    THETA0 = [0.4473222800422779, 0.3673667558902593, 0.1850675552332174]

########################################################################
@testset "   1. Конструирование модели                              " begin
########################################################################

    # --- @formula + VarEffect ------------------------------------------------
    lmm = refmodel()
    @test isa(lmm, Metida.LMM)
    @test Metida.isfitted(lmm) == false
    @test Metida.nobs(lmm) == 20
    @test Metida.rankx(lmm) == 6
    @test Metida.coefn(lmm) == 6
    @test Metida.thetalength(lmm) == 3
    @test Metida.fixedeffn(lmm) == 4
    @test Metida.responsename(lmm) == "var"
    @test Metida.nblocks(lmm) == 5
    @test coefnames(lmm) == ["(Intercept)", "sequence: 2", "period: 2",
                             "period: 3", "period: 4", "formulation: 2"]
    @test_nowarn formula(lmm)
    @test_nowarn show(io, lmm)                      # неподогнанная модель печатается

    # --- @lmmformula ---------------------------------------------------------
    lmmf = Metida.LMM(Metida.@lmmformula(var ~ sequence + period + formulation,
        random = formulation | subject : Metida.DIAG), df0)
    Metida.fit!(lmmf)
    @test Metida.m2logreml(lmmf) ≈ REML0 atol = 1E-6
    @test Metida.fixedeffn(lmmf) == 4

    # без интерсепта
    lmmf0 = Metida.fit(Metida.LMM, Metida.@lmmformula(var ~ 0 + sequence + period + formulation,
        random = formulation | subject : Metida.DIAG), df0)
    @test Metida.fixedeffn(lmmf0) == 3
    @test length(Metida.typeiii(lmmf0).name) == 3

    # функциональный терм в отклике
    lmmlog = Metida.fit(Metida.LMM, Metida.@lmmformula(log(var) ~ sequence + period + formulation,
        random = formulation | subject : Metida.DIAG), df0)
    @test Metida.responsename(lmmlog) == "log(var)"

    # --- fit(LMM, ...) как конструктор+подгонка ------------------------------
    lmmc = Metida.fit(Metida.LMM, @formula(var ~ sequence + period + formulation), df0;
        random = Metida.VarEffect(Metida.@covstr(formulation | subject), Metida.DIAG))
    @test Metida.m2logreml(lmmc) ≈ REML0 atol = 1E-6

    # --- Казуистика в спецификации субъекта ----------------------------------
    # subject = константный столбец, subject = литерал 1 и repeated|nosubj
    # должны давать один и тот же результат (один блок на все наблюдения).
    for subjspec in (Metida.@covstr(formulation | nosubj), Metida.@covstr(formulation | 1))
        l = Metida.LMM(@formula(var ~ sequence + period + formulation), df0;
            random = Metida.VarEffect(subjspec, Metida.DIAG))
        Metida.fit!(l)
        @test Metida.m2logreml(l) ≈ 25.129480634331067 atol = 1E-6
        @test Metida.nblocks(l) == 1
    end

    lrep = Metida.LMM(@formula(var ~ sequence + period + formulation), df0;
        repeated = Metida.VarEffect(Metida.@covstr(formulation | 1), Metida.DIAG))
    Metida.fit!(lrep)
    @test Metida.m2logreml(lrep) ≈ 25.00077786912235 atol = 1E-6

    lrep2 = Metida.LMM(@formula(var ~ sequence + period + formulation), df0;
        repeated = Metida.VarEffect(Metida.@covstr(formulation | nosubj)))
    Metida.fit!(lrep2)
    @test Metida.m2logreml(lrep2) ≈ 25.129480634331063 atol = 1E-6

    # --- Кодирование в случайной части ---------------------------------------
    # По умолчанию для терма с интерсептом и для 0+ ставится FullDummyCoding
    li = Metida.LMM(@formula(var ~ 1), df0;
        random = Metida.VarEffect(Metida.@covstr(1 + formulation | subject)))
    @test typeof(li.covstr.random[1].coding[:formulation]) <: StatsModels.FullDummyCoding

    lz = Metida.LMM(@formula(var ~ 1), df0;
        random = Metida.VarEffect(Metida.@covstr(0 + formulation | subject)))
    @test typeof(lz.covstr.random[1].coding[:formulation]) <: StatsModels.FullDummyCoding

    lc = Metida.LMM(@formula(var ~ 1), df0;
        random = Metida.VarEffect(Metida.@covstr(1 + formulation | subject),
                                  coding = Dict(:formulation => StatsModels.DummyCoding())))
    @test typeof(lc.covstr.random[1].coding[:formulation]) <: StatsModels.DummyCoding

    # --- Единственный фиксированный эффект -----------------------------------
    onefe = Metida.LMM(@formula(var ~ 1), df0;
        random = Metida.VarEffect(Metida.@covstr(formulation | subject), Metida.DIAG))
    @test coefnames(onefe) == ["(Intercept)"]    
    @test_nowarn show(io, onefe)

    # --- Вектор repeated-эффектов --------------------------------------------
    lmv = Metida.LMM(@formula(var ~ sequence + period + formulation), df0;
        repeated = [Metida.VarEffect(Metida.@covstr(formulation | subject), Metida.DIAG),
                    Metida.VarEffect(Metida.@covstr(1 | subject), Metida.SI)])
    Metida.fit!(lmv)
    @test Metida.isfitted(lmv)
    @test Metida.thetalength(lmv) == 3
    @test_nowarn show(io, lmv)

end # 1

########################################################################
@testset "   2. Подгонка: значения REML и ML                        " begin
########################################################################

    lmm = refmodel(); Metida.fit!(lmm)

    @test Metida.isfitted(lmm) == true
    @test Metida.m2logreml(lmm)                    ≈ REML0 atol = 1E-6
    @test Metida.m2logreml(lmm, Metida.theta(lmm)) ≈ REML0 atol = 1E-6
    @test Metida.m2logreml(lmm, THETA0)            ≈ REML0 atol = 1E-6
    @test Metida.m2logreml(lmm, [0.5, 0.3, 0.2])   ≈ 16.5746217198294 atol = 1E-6

    # Соотношения между вариантами критерия
    @test Metida.logreml(lmm) ≈ -0.5 * Metida.m2logreml(lmm) atol = 1E-10
    @test Metida.logreml(lmm) ≈ -8.120556322253035 atol = 1E-6
    @test Metida.logml(lmm)   ≈ -0.5 * Metida.m2logml(lmm)   atol = 1E-10

    @test Metida.m2logml(lmm)                                        ≈ 6.897520775993932 atol = 1E-6
    @test Metida.m2logml(lmm, coef(lmm))                             ≈ 6.897520775993932 atol = 1E-6
    @test Metida.m2logml(lmm, coef(lmm), Metida.theta(lmm); maxthreads = 8) ≈ 6.897520775993932 atol = 1E-6

    @test Metida.theta(lmm) ≈ THETA0 atol = 1E-6
    @test lmm.θ ≈ Metida.theta(lmm)                 # алиас через getproperty
    @test lmm.β ≈ coef(lmm)

    # Повторная подгонка идемпотентна
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm) ≈ REML0 atol = 1E-6
    # Подгонка с заданным init
    Metida.fit!(lmm; init = THETA0)
    @test Metida.m2logreml(lmm) ≈ REML0 atol = 1E-6
    # refitinit: старт от предыдущего решения
    Metida.fit!(lmm; refitinit = true)
    @test Metida.m2logreml(lmm) ≈ REML0 atol = 1E-6
    # hes = false не должен менять оценки
    lmmh = refmodel(); Metida.fit!(lmmh; hes = false)
    @test Metida.m2logreml(lmmh) ≈ REML0 atol = 1E-6

end # 2

########################################################################
@testset "   3. Внутренние величины REML                            " begin
########################################################################

    lmm = refmodel(); Metida.fit!(lmm)
    dv  = Metida.LMMDataViews(lmm)
    n   = length(lmm.covstr.vcovblock)

    reml, θs₂, remlθ₃, noerror =
        Metida.reml_sweep_β(lmm, dv, Metida.theta(lmm), Metida.coef(lmm))
    @test noerror == true
    @test reml   ≈ REML0 atol = 1E-6
    @test remlθ₃ ≈ 13.999999919958947 atol = 1E-6

    θ₁, θ₂, θ₃, noerror2 =
        Metida.core_sweep_β(lmm, dv, Metida.theta(lmm), Metida.coef(lmm), n)
    @test noerror2 == true
    @test θ₁ ≈ -43.860020472151916 atol = 1E-6
    @test θ₃ ≈ 13.999999919958947 atol = 1E-6

    # θ₂ хранится верхним треугольником; Symmetric даёт X'V⁻¹X
    @test istriu(θ₂)
    @test Symmetric(θ₂) ≈ θs₂ atol = 1E-6
    logdetθ₂ = logdet(Symmetric(θ₂))
    @test logdetθ₂ ≈ 20.370854266968205 atol = 1E-6

    # Разложение критерия: REML = Σlog|V| + log|X'V⁻¹X| + r'V⁻¹r + (N-p)log2π
    creml = (length(lmm.data.yv) - lmm.rankx) * log(2π)
    @test θ₁ + logdetθ₂ + θ₃ + creml ≈ REML0 atol = 1E-6
    # ML  = Σlog|V| + r'V⁻¹r + N log2π
    cml = length(lmm.data.yv) * log(2π)
    @test θ₁ + θ₃ + cml ≈ 6.897520775993932 atol = 1E-6

    # X'V⁻¹X должна быть положительно определена в точке оптимума
    @test isposdef(Symmetric(θ₂))
    # и согласована с vcov
    @test inv(Symmetric(θ₂)) ≈ vcov(lmm) atol = 1E-8

end # 3

########################################################################
@testset "   4. Интерфейс StatsAPI / StatsBase                      " begin
########################################################################

    lmm = refmodel(); Metida.fit!(lmm)

    @test isfitted(lmm) == true
    @test islinear(lmm) == true
    @test nobs(lmm) == 20
    @test dof_residual(lmm) == 14
    @test length(coef(lmm)) == 6
    @test length(stderror(lmm)) == 6
    @test size(vcov(lmm)) == (6, 6)
    @test vcov(lmm)[1, 1]   ≈ 0.11203611149231425 atol = 1E-6
    @test stderror(lmm)[1]  ≈ 0.33471795812641164 atol = 1E-6
    # stderror — корень из диагонали vcov
    @test stderror(lmm) ≈ sqrt.(diag(vcov(lmm))) atol = 1E-10
    @test issymmetric(Matrix(vcov(lmm)))

    @test length(modelmatrix(lmm)) == 120
    @test size(modelmatrix(lmm)) == (20, 6)
    @test isa(response(lmm), Vector)
    @test length(response(lmm)) == 20
    @test size(crossmodelmatrix(lmm), 1) == 6

    @test dof(lmm) == 9                              # текущее поведение, см. раздел 14
    @test aic(lmm)         ≈ 22.241112644506067 atol = 1E-6
    @test bic(lmm)         ≈ 24.158284633351833 atol = 1E-6
    @test aicc(lmm)        ≈ 24.64111264450606 atol = 1E-6
    @test Metida.caic(lmm) ≈ 27.158284633351833 atol = 1E-6
    # AIC = -2logREML + 2d, где d = число параметров ковариации
    @test aic(lmm) ≈ Metida.m2logreml(lmm) + 2 * Metida.thetalength(lmm) atol = 1E-8
    @test loglikelihood(lmm) ≈ Metida.logreml(lmm) atol = 1E-10

    ct = coeftable(lmm)
    @test length(ct.rownms) == 6
    @test_nowarn show(io, ct)

    # confint: три метода ddf, согласованность скалярной и векторной формы
    ci_all = Metida.confint(lmm)
    @test length(ci_all) == 6
    @test ci_all[end][1] ≈ -0.7630380758015894 atol = 1E-4
    @test Metida.confint(lmm, 6)[1] ≈ ci_all[end][1] atol = 1E-10
    @test Metida.confint(lmm; ddf = :residual)[end][1] ≈ -0.6740837049617738 atol = 1E-4
    @test_nowarn Metida.confint(lmm; ddf = :contain)             # SPSS:7
    # нижняя граница строго меньше верхней, оценка внутри интервала
    for (i, c) in enumerate(ci_all)
        @test c[1] < c[2]
        @test c[1] < coef(lmm)[i] < c[2]
    end
    # более широкий уровень даёт более широкий интервал
    c90 = Metida.confint(lmm; level = 0.90)
    c99 = Metida.confint(lmm; level = 0.99)
    @test c99[1][2] - c99[1][1] > c90[1][2] - c90[1][1]

end # 4

########################################################################
@testset "   5. Матрицы модели                                      " begin
########################################################################

    lmm = refmodel(); Metida.fit!(lmm)

    G = Metida.gmatrix(lmm, 1)
    R = Metida.rmatrix(lmm, 1)
    V = Metida.vmatrix(lmm, 1)
    @test sum(G) ≈ 0.3350555603325126 atol = 1E-6
    @test sum(R) ≈ 0.13699999248885292 atol = 1E-6
    @test sum(V) ≈ 1.4772222338189034 atol = 1E-6
    @test Metida.gmatrixipd(lmm)                       # G положительно определена
    @test issymmetric(Matrix(V))
    @test isposdef(Symmetric(Matrix(V)))
    @test size(V, 1) == length(lmm.covstr.vcovblock[1])

    # V(θ) через явный вектор параметров совпадает с V подогнанной модели
    @test Metida.vmatrix(Metida.theta(lmm), lmm, 1) ≈ V atol = 1E-10

    # vmatrix! пишет в предоставленный буфер
    Vb = zeros(size(V))
    Metida.vmatrix!(Vb, Metida.theta(lmm), lmm, 1)
    @test Symmetric(Vb) ≈ Symmetric(Matrix(V)) atol = 1E-10

    # Все блоки собираются без ошибок и положительно определены
    for i in 1:Metida.nblocks(lmm)
        Vi = Metida.vmatrix(lmm, i)
        @test isposdef(Symmetric(Matrix(Vi)))
    end

    @test size(Metida.reml_hessian(lmm)) == (3, 3)
    @test sum(Metida.reml_hessian(lmm)) ≈ 1118.160713481362 atol = 1E-2
    @test issymmetric(round.(Metida.reml_hessian(lmm); digits = 6))

    # Случайные эффекты по блокам
    @test Metida.raneffn(lmm) == 1
    for i in 1:Metida.nblocks(lmm)
        @test_nowarn Metida.raneff(lmm, i)
    end
    # Для модели без случайной части raneff == nothing
    lmmr = Metida.LMM(@formula(var ~ sequence + period + formulation), df0;
        repeated = Metida.VarEffect(Metida.@covstr(formulation | subject), Metida.DIAG))
    Metida.fit!(lmmr)
    @test Metida.raneffn(lmmr) == 0
    @test Metida.raneff(lmmr, 1) === nothing

end # 5

########################################################################
@testset "   6. Инференс: DF, контрасты, тип III                    " begin
########################################################################

    lmm = refmodel(); Metida.fit!(lmm)
    l3  = [0 0 1 0 0 0; 0 0 0 1 0 0; 0 0 0 0 1 0]      # контраст на period

    # --- Satterthwaite: три формы вызова должны совпадать --------------------
    d_int = Metida.dof_satter(lmm, 6)
    d_vec = Metida.dof_satter(lmm, [0, 0, 0, 0, 0, 1])
    d_all = Metida.dof_satter(lmm)
    @test d_int ≈ 5.81896814947982 atol = 1E-2          # SPSS:1
    @test d_vec ≈ d_int atol = 1E-6
    @test d_all[end] ≈ d_int atol = 1E-6
    @test length(d_all) == Metida.coefn(lmm)
    @test all(x -> x >= 1.0, d_all)
    @test all(x -> x <= dof_residual(lmm), d_all)

    # --- Матричный контраст --------------------------------------------------
    @test Metida.dof_satter(lmm, l3) ≈ 7.575447546211385 atol = 1E-2   # SPSS:2
    @test Metida.dof_satter(lmm, Metida.lcontrast(lmm, 3)) ≈
          Metida.dof_satter(lmm, l3) atol = 1E-6
    @test Metida.fvalue(lmm, l3) ≈ 0.202727915619993 atol = 1E-2

    # lcontrast: размерность и ранг
    L3 = Metida.lcontrast(lmm, 3)
    @test size(L3, 2) == Metida.coefn(lmm)
    @test rank(L3) == 3
    @test_nowarn Metida.lcontrast(lmm, 1)
    @test_nowarn Metida.lcontrast(lmm, 4)
    @test_throws ErrorException Metida.lcontrast(lmm, 99)

    # --- typeiii: три метода ddf --------------------------------------------
    t3 = Metida.typeiii(lmm)
    @test length(t3.name) == 4
    @test all(x -> x >= 0, t3.f)
    @test all(x -> 0 <= x <= 1, t3.pval)
    @test t3.pval[4] ≈ 0.7852154468081014 atol = 1E-6                  # SPSS:3
    @test_nowarn Metida.typeiii(lmm; ddf = :residual)
    @test_nowarn Metida.typeiii(lmm; ddf = :contain)                   # SPSS:8
    @test_nowarn show(io, t3)

    # typeiii и contrast на одном и том же контрасте дают одно p-значение
    ct = Metida.contrast(lmm, l3)
    @test t3.pval[3] ≈ ct.pval[1] atol = 1E-8
    @test ct.ndf[1] ≈ 3.0
    @test_nowarn show(io, ct)
    @test_nowarn Metida.contrast(lmm, l3; name = "period", ddf = :residual)
    # прямое задание ddf числом
    ctd = Metida.contrast(lmm, l3; ddf = 14)
    @test ctd.df[1] == 14

    # --- estimate ------------------------------------------------------------
    e1 = Metida.estimate(lmm, [0, 0, 0, 0, 0, 1]; level = 0.9)
    @test_nowarn show(io, e1)
    eall = Metida.estimate(lmm; level = 0.9)
    @test_nowarn show(io, eall)

    # --- dof_contain ---------------------------------------------------------
    lmmc = Metida.LMM(@formula(var ~ period * formulation), df0;
        random = Metida.VarEffect(Metida.@covstr(formulation + sequence | nosubj), Metida.SI))
    Metida.fit!(lmmc)
    @test Metida.m2logreml(lmmc, [0.222283, 0.444566]) ≈ Metida.m2logreml(lmmc) atol = 1E-6
    @test Metida.dof_contain(lmmc, 1) == 12                            # SPSS:9
    @test Metida.dof_contain(lmmc, 5) == 8                             # SPSS:9

    tt = Metida.typeiii(lmmc)
    @test tt.f[2]    ≈ 0.185268 atol = 1E-5                            # SPSS:4
    @test tt.ndf[2]  ≈ 3.0      atol = 1E-5
    @test tt.df[2]   ≈ 3.39086  atol = 1E-5                            # SPSS:4
    @test tt.pval[2] ≈ 0.900636 atol = 1E-5
    # Диагностика на случай платформенного расхождения (см. SPSS_VALIDATION.md):
    # df == 1.0 означает срабатывание заглушки df < 1 в dof_satter_, а не
    # законную оценку — при lclr = 3 значения из (1, 2] недостижимы.
    @test tt.df[2] > 2.0

end # 6

########################################################################
@testset "   7. Link-функции и стратегии первого шага               " begin
########################################################################

    mk() = Metida.LMM(@formula(var ~ sequence + period + formulation), df0;
        random = Metida.VarEffect(Metida.@covstr(1 + formulation | subject), Metida.CSH;
                                  coding = Dict(:formulation => StatsModels.DummyCoding())))

    l = mk(); Metida.fit!(l; rholinkf = :sqsigm)
    @test Metida.m2logreml(l) ≈ 10.314822559210157 atol = 1E-6
    @test Metida.dof_satter(l, [0, 0, 0, 0, 0, 1]) ≈ 6.043195705464293 atol = 1E-2

    l = mk(); Metida.fit!(l; rholinkf = :atan)
    @test Metida.m2logreml(l) ≈ 10.314837309793571 atol = 1E-6

    l = mk(); Metida.fit!(l; rholinkf = :psigm)
    @test Metida.m2logreml(l) ≈ 10.86212458333098 atol = 1E-6

    l = mk(); Metida.fit!(l; varlinkf = :sq)
    @test Metida.m2logreml(l) ≈ 10.314822479530243 atol = 1E-6

    # --- aifirst -------------------------------------------------------------
    lmm = refmodel()
    Metida.fit!(lmm; aifirst = :score)
    @test Metida.m2logreml(lmm) ≈ REML0 atol = 1E-6
    Metida.fit!(lmm; aifirst = :ai)
    @test Metida.m2logreml(lmm) ≈ REML0 atol = 1E-6
    Metida.fit!(lmm; aifirst = :ai, init = THETA0)
    @test Metida.m2logreml(lmm) ≈ REML0 atol = 1E-6
    Metida.fit!(lmm; aifirst = :default)
    @test Metida.m2logreml(lmm) ≈ REML0 atol = 1E-6

    # --- альтернативные оптимизаторы -----------------------------------------
    lmm = refmodel(); Metida.fit!(lmm; optmethod = Metida.LBFGS_OM)
    @test Metida.m2logreml(lmm) ≈ REML0 atol = 1E-4
    lmm = refmodel(); Metida.fit!(lmm; optmethod = Metida.BFGS_OM)
    @test Metida.m2logreml(lmm) ≈ REML0 atol = 1E-4

end # 7

########################################################################
@testset "   8. Потоки и воспроизводимость                          " begin
########################################################################

    # Однопоточный и многопоточный прогон должны совпадать в пределах
    # накопления ошибки суммирования по блокам.
    lmm1 = refmodel(); Metida.fit!(lmm1; maxthreads = 1)
    lmm4 = refmodel(); Metida.fit!(lmm4; maxthreads = 4)
    @test Metida.m2logreml(lmm1) ≈ REML0 atol = 1E-6
    @test Metida.m2logreml(lmm4) ≈ Metida.m2logreml(lmm1) atol = 1E-6
    @test Metida.theta(lmm1) ≈ Metida.theta(lmm4) atol = 1E-6
    @test Metida.dof_satter(lmm1, 6) ≈ Metida.dof_satter(lmm4, 6) atol = 1E-4

    # Модель с блоками разного размера (BE-подобная)
    be = Metida.LMM(@formula(var ~ sequence + period + formulation), df0;
        random   = Metida.VarEffect(Metida.@covstr(formulation | subject), Metida.CSH),
        repeated = Metida.VarEffect(Metida.@covstr(formulation | subject), Metida.DIAG))
    Metida.fit!(be; aifirst = :score)
    @test Metida.m2logreml(be) ≈ 10.065238626765524 atol = 1E-6
    Metida.fit!(be; maxthreads = 1)
    @test Metida.m2logreml(be) ≈ 10.065238626765524 atol = 1E-6

end # 8

########################################################################
@testset "   9. Веса наблюдений                                     " begin
########################################################################

    # Постоянные веса 0.5: масштаб V меняется, но REML инвариантен,
    # т.к. масштаб поглощается оценкой дисперсии.
    lw = Metida.LMM(@formula(var ~ sequence + period + formulation), df0;
        random = Metida.VarEffect(Metida.@covstr(formulation | subject), Metida.DIAG),
        wts = df0.wtsc)
    Metida.fit!(lw)
    @test Metida.m2logreml(lw) ≈ REML0 atol = 1E-6

    # Веса из колонки: по имени и по символу — идентично
    lws = Metida.LMM(@formula(var ~ sequence + period + formulation), df0;
        random = Metida.VarEffect(Metida.@covstr(formulation | subject), Metida.DIAG),
        wts = "wts")
    Metida.fit!(lws)
    @test Metida.m2logreml(lws) ≈ 17.823729 atol = 1E-6                # SPSS:5 (сверено, SPSS 28)

    lwy = Metida.LMM(@formula(var ~ sequence + period + formulation), df0;
        random = Metida.VarEffect(Metida.@covstr(formulation | subject), Metida.DIAG),
        wts = :wts)
    Metida.fit!(lwy)
    @test Metida.m2logreml(lwy) ≈ Metida.m2logreml(lws) atol = 1E-8

    # Неверная длина вектора весов — предупреждение, веса игнорируются
    @test_warn "wts count not equal observations count! wts not used." begin
        lbad = Metida.LMM(@formula(var ~ sequence + period + formulation), df0;
            random = Metida.VarEffect(Metida.@covstr(formulation | subject), Metida.DIAG),
            wts = ones(10))
        Metida.fit!(lbad)
        @test Metida.m2logreml(lbad) ≈ REML0 atol = 1E-6
    end

    # Матричные веса
    lmw = Metida.LMM(@formula(var ~ sequence + period + formulation), df0;
        random = Metida.VarEffect(Metida.@covstr(formulation | subject), Metida.DIAG),
        wts = matwts)
    @test_nowarn Metida.fit!(lmw)
    @test Metida.isfitted(lmw)

    # Экспериментальная взвешенная ковариация SWC
    lswc = Metida.LMM(@formula(var ~ sequence + period + formulation), df0;
        repeated = Metida.VarEffect(Metida.@covstr(1 | subject), Metida.SWC(matwts)))
    @test_nowarn Metida.fit!(lswc)
    @test_nowarn show(io, lswc)

end # 9

########################################################################
@testset "  10. Краевые случаи                                      " begin
########################################################################

    # --- Пропуски в отклике --------------------------------------------------
    lmiss = Metida.LMM(@formula(var ~ sequence + period + formulation), df0m;
        random = Metida.VarEffect(Metida.@covstr(formulation | subject), Metida.DIAG))
    Metida.fit!(lmiss)
    @test Metida.m2logreml(lmiss) ≈ 16.636012616466203 atol = 1E-6
    @test nobs(lmiss) < size(df0m, 1)

    # --- Неполные данные внутри субъекта (df1) -------------------------------
    linc = Metida.LMM(@formula(var ~ sequence + period + formulation), df1;
        random   = Metida.VarEffect(Metida.@covstr(formulation | subject), Metida.CSH),
        repeated = Metida.VarEffect(Metida.@covstr(formulation | subject), Metida.DIAG))
    Metida.fit!(linc; hes = false)
    @test Metida.m2logreml(linc) ≈ 14.819463206995163 atol = 1E-6
    @test Metida.dof_satter(linc, 6) ≈ 3.7026122766034915 atol = 1E-2    # SPSS:6

    # --- Целочисленный отклик + функциональный терм в случайной части --------
    lmmint = @test_warn "Response variable not <: AbstractFloat" Metida.fit(Metida.LMM,
        Metida.@lmmformula(varint ~ formulation, random = 1 + var^2 | subject : Metida.SI), df0)
    Metida.fit!(lmmint)
    @test Metida.m2logreml(lmmint) ≈ 84.23373276096902 atol = 1E-6

    # --- Неполный ранг X -----------------------------------------------------
    lrd = @test_warn "Fixed-effect matrix not full-rank" Metida.LMM(
        @formula(lnpk ~ sequence + period + treatment + subject), dfrdsfda;
        random = Metida.VarEffect(Metida.@covstr(treatment | subject), Metida.DIAG))
    @test Metida.rankx(lrd) < Metida.coefn(lrd)
    # API на неподогнанной модели не должен падать
    @test_nowarn Metida.coef(lrd)
    @test_nowarn Metida.vcov(lrd)
    @test_nowarn Metida.stderror(lrd)
    @test_nowarn Metida.fit!(lrd)
    @test_nowarn Metida.confint(lrd; level = 0.95, ddf = :satter)
    @test_nowarn Metida.lcontrast(lrd, 5)
    @test_nowarn Metida.typeiii(lrd)
    @test_nowarn show(io, lrd)
    # Отброшенные коэффициенты помечены NaN в vcov/stderror
    @test any(isnan, Metida.stderror(lrd))
    @test length(Metida.coef(lrd)) == Metida.coefn(lrd)

end # 10

########################################################################
@testset "  11. Логирование и отображение                           " begin
########################################################################

    lmm = refmodel(); Metida.fit!(lmm)
    @test Metida.msgnum(lmm.log) == 3
    @test Metida.msgnum(lmm.log, :INFO) >= 1
    @test isa(Metida.getlog(lmm), Vector)
    @test_nowarn show(io, lmm)
    @test_nowarn show(io, Metida.getlog(lmm))

    # Лог очищается при повторной подгонке
    n1 = Metida.msgnum(lmm.log)
    Metida.fit!(lmm)
    @test Metida.msgnum(lmm.log) == n1

    # Отображение всех типов ковариации, представленных в API
    for ct in (Metida.SI, Metida.DIAG, Metida.CS, Metida.CSH,
               Metida.AR, Metida.ARH, Metida.ARMA, Metida.TOEP, Metida.TOEPH, Metida.UN)
        @test_nowarn show(io, ct)
    end
    for ctf in (Metida.TOEPP(2), Metida.TOEPHP(2))
        @test_nowarn show(io, ctf)
    end
    @test_nowarn show(io, Metida.VarEffect(Metida.@covstr(formulation | subject), Metida.DIAG))

end # 11

########################################################################
@testset "  12. Обработка ошибок                                    " begin
########################################################################

    lmm = refmodel()
    # Инференс на неподогнанной модели
    @test_throws ErrorException Metida.typeiii(lmm)
    @test_throws ErrorException Metida.dof_satter(lmm, 1)
    @test_throws ErrorException Metida.contrast(lmm, [0 0 0 0 0 1])

    Metida.fit!(lmm)
    # Несогласованная размерность контраста
    @test_throws ErrorException Metida.contrast(lmm, [0 0 1])
    @test_throws ErrorException Metida.lcontrast(lmm, 0)
    # Неизвестный метод / метрика
    @test_throws ErrorException Metida.dof_satter(lmm, [0 0 1])

end # 12

########################################################################
@testset "  13. Симуляция и ресемплинг                              " begin
########################################################################

    lmm = refmodel(); Metida.fit!(lmm)
    rng = StableRNG(20240502)

    # rand по подогнанной модели
    y1 = Metida.rand(rng, lmm)
    @test length(y1) == nobs(lmm)
    @test all(isfinite, y1)
    y2 = Metida.rand(rng, lmm, Metida.theta(lmm))
    @test length(y2) == nobs(lmm)
    y3 = Metida.rand(rng, lmm, Metida.theta(lmm), Metida.coef(lmm))
    @test length(y3) == nobs(lmm)
    # rand! в предоставленный буфер
    buf = zeros(nobs(lmm))
    Metida.rand!(rng, buf, lmm)
    @test all(isfinite, buf)
    # Воспроизводимость по seed
    @test Metida.rand(StableRNG(1), lmm) == Metida.rand(StableRNG(1), lmm)

    # Параметрический бутстрап (малое n — только смоук)
    br = Metida.bootstrap(lmm; n = 20, verbose = false, rng = StableRNG(20240503))
    @test_nowarn show(io, br)
    cb = Metida.confint(br; level = 0.95, method = :bp)
    @test length(cb) == Metida.coefn(lmm)
    for c in cb
        @test c[1] < c[2]
    end
    for m in (:bp, :rbp, :norm, :bcnorm)
        @test_nowarn Metida.confint(br, 1; method = m)
    end
    @test_throws ErrorException Metida.confint(br, 1; method = :nonexistent)
    @test_throws ErrorException Metida.confint(br, 1; metric = :nonexistent)

end # 13


end # Publick API basic tests
#=
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
    @test Metida.logreml(lmm) ≈ -0.5*16.241112644506067 atol=1E-6

    @test Metida.m2logreml(lmm, [0.447322, 0.367367, 0.185068]) ≈ 16.241112644506067 atol=1E-6
    @test Metida.m2logreml(lmm, [0.5, 0.3, 0.2]) ≈ 16.5746217198294 atol=1E-6
    
    # core_sweep_β REML, ML estimate test; 
    reml, θs₂, remlθ₃, noerror = Metida.reml_sweep_β(lmm, Metida.LMMDataViews(lmm), Metida.theta(lmm), Metida.coef(lmm))
    @test reml ≈ 16.241112644506067 atol=1E-6
    @test θs₂ ≈ [55.8946  33.5368    13.4807   14.4666    13.4807   32.8767
    33.5368  33.5368     6.90537   9.86301    6.90537  19.726
    13.4807   6.90537   79.7331    0.0      -66.2523    6.57534
    14.4666   9.86301    0.0      80.226      0.0       9.86301
    13.4807   6.90537  -66.2523    0.0       79.7331    6.57534
    32.8767  19.726      6.57534   9.86301    6.57534  32.8767] atol=1E-3
    @test remlθ₃ ≈ 13.999999919958947 atol=1E-6

    θ₁, θ₂, θ₃, noerror = Metida.core_sweep_β(lmm, Metida.LMMDataViews(lmm), Metida.theta(lmm), Metida.coef(lmm), length(lmm.covstr.vcovblock))
    @test noerror == true
    c             = (length(lmm.data.yv) - lmm.rankx)*log(2π)
    @test θ₁ ≈  -43.860020472151916 atol=1E-6
    @test θ₂ ≈ [55.8946  33.5368  13.4807   14.4666    13.4807   32.8767
    0.0     33.5368   6.90537   9.86301    6.90537  19.726
    0.0      0.0     79.7331    0.0      -66.2523    6.57534
    0.0      0.0      0.0      80.226      0.0       9.86301
    0.0      0.0      0.0       0.0       79.7331    6.57534
    0.0      0.0      0.0       0.0        0.0      32.8767] atol=1E-3
    @test θ₃ ≈ 13.999999919958947 atol=1E-6
    logdetθ₂ = logdet(Symmetric(θ₂))
    @test logdetθ₂  ≈ 20.370854266968205 atol=1E-6
    @test θ₁ + logdetθ₂ + θ₃ + c ≈ 16.241112644506067 atol=1E-6
    # ML test
    c = length(lmm.data.yv)*log(2π)
    @test θ₁ + θ₃ + c ≈ 6.897520775993932 atol=1E-6
    @test  Metida.m2logml(lmm, coef(lmm), Metida.theta(lmm); maxthreads = 8) ≈ 6.897520775993932
    @test  Metida.m2logml(lmm, coef(lmm)) ≈ 6.897520775993932
    @test  Metida.m2logml(lmm) ≈ 6.897520775993932
    @test  Metida.logml(lmm) ≈ -0.5*6.897520775993932
    #
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
    @test sum(Metida.reml_hessian(lmm))    ≈ 1118.160713481362 atol=1E-2
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
    @test coefnames(onefelmm) == ["(Intercept)"]
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
    # @test Metida.dof_satter(lmm, 6)   ≈ 3.981102548214154 atol=1E-2 after reml_hessian 
    @test Metida.dof_satter(lmm, 6)   ≈ 3.702612265174825 atol=1E-6

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

    # Intercept term in random part
    lmm = Metida.LMM(@formula(var~1), df0;
    random = Metida.VarEffect(Metida.@covstr(1+formulation|subject)))
    @test typeof( lmm.covstr.random[1].coding[:formulation]) <: StatsModels.FullDummyCoding
    # Zero term in random part
    lmm = Metida.LMM(@formula(var~1), df0;
    random = Metida.VarEffect(Metida.@covstr(0+formulation|subject)))
    @test typeof( lmm.covstr.random[1].coding[:formulation]) <: StatsModels.FullDummyCoding
    # Intercept term in random part and coding
    lmm = Metida.LMM(@formula(var~1), df0;
    random = Metida.VarEffect(Metida.@covstr(1+formulation|subject), coding = Dict(:formulation => StatsModels.DummyCoding())))
    @test typeof( lmm.covstr.random[1].coding[:formulation]) <: StatsModels.DummyCoding

    # Rank defficient

    lmm = Metida.LMM(@formula(lnpk~sequence+period+treatment+subject), dfrdsfda;
    random = Metida.VarEffect(Metida.@covstr(treatment|subject), Metida.DIAG))
    @test_nowarn Metida.coef(lmm)
    @test_nowarn Metida.vcov(lmm)
    @test_nowarn Metida.stderror(lmm)
    @test_nowarn Metida.fit!(lmm)
    @test_nowarn Metida.confint(lmm; level=0.95, ddf = :satter)
    @test_nowarn Metida.lcontrast(lmm, 5)
    @test_nowarn Metida.typeiii(lmm)
    @test_nowarn show(io, lmm)

end
=#
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
    @test Metida.m2logreml(lmm)  ≈ 697.2241355154041 atol=1E-6

    lmmf = Metida.@lmmformula(response ~ 1 + factor,
    random = 1|subject/r1,
    repeated = p|subject:Metida.CSH)

    lmm = Metida.LMM(lmmf,
    ftdf3; contrasts=Dict(:factor => DummyCoding(; base=1.0)))

    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm)  ≈ 697.2241355154041 atol=1E-6
    io = IOBuffer();
    @test_nowarn show(io, lmmf)
    @test Metida.dof_satter(lmm)[2] ≈ 21.944281700360293 atol=1E-6
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
    @test Metida.m2logreml(lmm)  ≈ 698.8792511057682 atol=1E-6
    #SPSS 22.313
    @test Metida.dof_satter(lmm)[2] ≈ 22.31337200822804 atol=1E-6
    #SPSS 
    re = Metida.raneff(lmm, 1)
    #= 
    # REVALIDATE!
    @test re[1][1][2][1] ≈ 2.147751 atol=1E-5
    @test re[1][1][2][2] ≈ 1.446182 atol=1E-5
    @test re[1][1][2][3] ≈ 1.496007 atol=1E-5
    =#
    @test re[1][1][2][1] ≈ 5.482215030174481 atol=1E-5
    @test re[1][1][2][2] ≈ 7.921075494820502 atol=1E-5
    @test re[1][1][2][3] ≈ 4.24853688385001 atol=1E-5
end

@testset "  Model: ARMA/SI                                           " begin
    io = IOBuffer();
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3; contrasts=Dict(:factor => DummyCoding(; base=1.0)),
    random = Metida.VarEffect(Metida.@covstr(p|r1&r2), Metida.ARMA),
    )
    Metida.fit!(lmm; verbose = 3, io = io)
    #[1.2964e-5, 0.0299594, 0.0699728, 3.69557]
    println(io, lmm.log)
    @test Metida.m2logreml(lmm)  ≈ 913.9176298311813 atol=1E-6
    #SPSS 166
    @test Metida.dof_satter(lmm)[2] ≈ 165.99999999999005 atol=1E-6
end

@testset "  Model: ARH/SI (subjects with &)                          " begin
    # SPSS 707.377
    # nowarn
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3; contrasts=Dict(:factor => DummyCoding(; base=1.0)),
    random = Metida.VarEffect(Metida.@covstr(r1|s2&factor), Metida.ARH),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm)  ≈ 707.3765873864152 atol=1E-6
    #SPSS 23.093
    @test Metida.dof_satter(lmm, [0, 1]) ≈ 23.093021655996232 atol=1E-6

    #SPSS 691.360073
    lmm = Metida.LMM(@formula(nrhoresp ~ 1 + factor), ftdf3; contrasts=Dict(:factor => DummyCoding(; base=1.0)),
    random = Metida.VarEffect(Metida.@covstr(r1|s2&factor), Metida.ARH),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm)  ≈ 691.3600726310308 atol=1E-6
    mtt = Metida.typeiii(lmm)
    #SPSS 48.550474
    @test mtt.df[2] ≈ 48.550473645995346 atol=1E-6

end
@testset "  Model: INT, *, DIAG/SI                                   " begin
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3;
    random = Metida.VarEffect(Metida.@covstr(1 + r2 * r1|subject), Metida.DIAG; coding=Dict(:r1 => DummyCoding(), :r2 => DummyCoding()))
    )
    Metida.fit!(lmm)
    @test Metida.theta(lmm)  ≈ [2.796694409004289, 2.900485570555582, 3.354913215348968, 2.0436114769223237, 1.8477830405766895, 2.0436115732330955, 1.0131934233937254] atol=1E-5 # atol=1E-8 !
    @test Metida.m2logreml(lmm)  ≈ 713.0655862252027 atol=1E-6
end
@testset "  Model: &, DIAG/SI                                        " begin
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3;
    random = Metida.VarEffect(Metida.@covstr(r1&r2|subject), Metida.DIAG),
    )
    Metida.fit!(lmm)
    @test Metida.theta(lmm)  ≈ [3.0325005960015985, 3.343826588448401, 1.8477830405766895, 1.8477830405766895, 1.8477830405766895, 4.462942536844632, 1.0082345219318216] atol=1E-5 # atol=1E-8 !
    @test Metida.m2logreml(lmm)  ≈ 719.9413776641368 atol=1E-6
end
@testset "  Model: INT, +,  TOEPHP(3)/SI                             " begin
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3;
    random = Metida.VarEffect(Metida.@covstr(1 + r1 + r2|subject), Metida.TOEPHP(3); coding = Dict(:r1 => DummyCoding(), :r2 => DummyCoding())),
    )
    Metida.fit!(lmm)
    @test Metida.theta(lmm)  ≈ [2.843269324925114, 3.3598654954863423, 7.582560427911907e-10, 4.133572859333964, -0.24881591201506625, 0.46067672264107506, 1.0091887333170306] atol=1E-8
    @test Metida.m2logreml(lmm)  ≈ 705.9946274598822 atol=1E-6
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
    @test Metida.m2logreml(lmm)  ≈ 710.1998669150806 atol=1E-6
end
@testset "  Model: TOEPP(2)/SI                                       " begin
    # nowarn
    io = IOBuffer();
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3;
    random = Metida.VarEffect(Metida.@covstr(r1|subject), Metida.TOEPP(2)),
    )
    Metida.fit!(lmm)
    Base.show(io, lmm)
    @test Metida.m2logreml(lmm)  ≈ 715.2410264030134 atol=1E-6
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
    @test Metida.m2logreml(lmm)  ≈ 773.9575538254085 atol=1E-6
end
@testset "  Model: TOEPH/SI                                          " begin
    # nowarn
    io = IOBuffer();
    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3;
    random = Metida.VarEffect(Metida.@covstr(r1|subject), Metida.TOEPH),
    )
    Metida.fit!(lmm)
    Base.show(io, lmm)
    @test Metida.m2logreml(lmm)  ≈ 705.7916833009426 atol=1E-6
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
    @test Metida.m2logreml(lmm)  ≈ 713.5850978377632 atol=1E-6
end

@testset "  Model: UN (repeated) missing                             " begin
    io = IOBuffer();
    lmm = Metida.LMM(@formula(val~1), rep_missing, repeated = Metida.VarEffect(Metida.@covstr(group|id), Metida.UN))
    fit!(lmm)
    #SPSS check +
    @test Metida.m2logreml(lmm) ≈ 138.674280 atol=1E-6
    @test  Metida.theta(lmm)[1]^2 ≈ 0.930862 atol=1E-6
    @test  Metida.theta(lmm)[4]^2 ≈ 13.833087 atol=1E-6
    @test  Metida.theta(lmm)[5] ≈ 0.222058 atol=1E-6
    @test  Metida.theta(lmm)[9] ≈ -0.516841 atol=1E-6
    Base.show(io, lmm)

    mlmm = Metida.LMM(@formula(mval~1), rep_missing, repeated = Metida.VarEffect(Metida.@covstr(group|id), Metida.UN))
    fit!(mlmm)
    #SPSS check +
    @test Metida.m2logreml(mlmm) ≈ 103.523394 atol=1E-6
    @test  Metida.theta(mlmm)[1]^2 ≈ 0.815214 atol=1E-6
    @test  Metida.theta(mlmm)[4]^2 ≈ 13.881296 atol=1E-6
    @test  Metida.theta(mlmm)[5] ≈ 0.295511 atol=1E-6
    @test  Metida.theta(mlmm)[9] ≈ -0.127460 atol=1E-6
    Base.show(io, mlmm)
    Base.show(io, mlmm.log)
end

@testset "  Model: BE RDS 1, FDA model                               "  begin
    
    lmm = Metida.LMM(@formula(lnpk~sequence+period+treatment), dfrdsfda;
    random = Metida.VarEffect(Metida.@covstr(treatment|subject), Metida.CSH),
    repeated = Metida.VarEffect(Metida.@covstr(treatment|subject), Metida.DIAG),
    )
    Metida.fit!(lmm)
    @test collect(Metida.confint(lmm)[6]) ≈  [0.053789444388152474, 0.23713911100102136] atol=1E-6
    anovatable = Metida.typeiii(lmm)
    @test anovatable.pval ≈ [3.087934998046721e-63, 0.9176105002577626, 0.6522549061162943, 0.002010933915677479] atol=1E-4

    est = Metida.estimate(lmm, [0,0,0,0,0,1]; level = 0.9)
    @test est.t[1] ≈ 3.12818 atol=1E-4
    @test est.pval[1] ≈ 0.0020 atol=1E-4
    @test est.cil[1] ≈ 0.06863 atol=1E-4
    @test est.ciu[1] ≈ 0.2223 atol=1E-4

    lmm = Metida.LMM(@formula(lnpk~0+sequence+period+treatment), dfrdsfda;
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
    @test Metida.m2logreml(lmm)  ≈ -1.0745407333692825 atol=1E-6

    # Unstructured
    lmm = Metida.LMM(@formula(log(Var)~Sequence+Period+Formulation), dfrds;
    repeated = Metida.VarEffect(Metida.@covstr(Formulation|Subject), Metida.UN),
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm)  ≈ -3.895979534278979 atol=1E-6


    lmm2 = Metida.LMM(@formula(log(Var)~Sequence+Period+Formulation), dfrds;
    repeated = Metida.VarEffect(Metida.@covstr(Formulation|Subject), Metida.CSH),
    )
    Metida.fit!(lmm2)

    @test Metida.m2logreml(lmm)  ≈ Metida.m2logreml(lmm2) atol=1E-6
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

@testset "  Model: Augmented covariance                              " begin
    lmm1 = Metida.LMM(@formula(response ~ 1), ftdf3;
    repeated = [Metida.VarEffect(Metida.@covstr(r1|subject), Metida.DIAG), Metida.VarEffect(Metida.@covstr(1|subject), Metida.ACOV(Metida.AR))]
    )
    Metida.fit!(lmm1)

    lmm2 = Metida.LMM(@formula(response ~ 1), ftdf3;
    repeated = [Metida.VarEffect(Metida.@covstr(r1|subject), Metida.ARH)]
    )
    Metida.fit!(lmm2)
    @test Metida.m2logreml(lmm1)  ≈ Metida.m2logreml(lmm2) 


    lmm1 = Metida.LMM(@formula(response ~ 1), ftdf3;
    repeated = [Metida.VarEffect(Metida.@covstr(r1|subject), Metida.DIAG), Metida.VarEffect(Metida.@covstr(1|subject), Metida.ACOV(Metida.CS))]
    )
    Metida.fit!(lmm1)

    lmm2 = Metida.LMM(@formula(response ~ 1), ftdf3;
    repeated = [Metida.VarEffect(Metida.@covstr(r1|subject), Metida.CSH)]
    )
    Metida.fit!(lmm2)
    @test Metida.m2logreml(lmm1)  ≈ Metida.m2logreml(lmm2) 

    lmm = Metida.LMM(@formula(response ~ 1), ftdf3;
    repeated = [Metida.VarEffect(Metida.@covstr(r1|subject), Metida.DIAG), Metida.VarEffect(Metida.@covstr(1|subject&r2), Metida.ACOV(Metida.AR))]
    )
    Metida.fit!(lmm)
    @test Metida.m2logreml(lmm)  ≈ 800.7606945922405 atol=1E-6


    lmm1 = Metida.LMM(@formula(resp ~ 0 + device), devday;
    repeated = [Metida.VarEffect(Metida.@covstr(device|subj&day), Metida.UN), 
    Metida.VarEffect(Metida.@covstr(1|subj&device), Metida.ACOV(Metida.CS))]
    )
    Metida.fit!(lmm1)

    lmm2 = Metida.LMM(@formula(resp ~ 0 + device), devday;
    repeated = [Metida.VarEffect(Metida.@covstr(device|subj&day), Metida.UN), 
    Metida.VarEffect(Metida.@covstr(1|subj&device), Metida.ACOV(Metida.AR))]
    )
    Metida.fit!(lmm2)
    @test Metida.m2logreml(lmm1)  ≈ Metida.m2logreml(lmm2) 

    io = IOBuffer();

    @test_nowarn show(io, lmm2)
end



@testset "  Random                                                   " begin
    lmm = Metida.LMM(@formula(response ~ 1 + factor*time), ftdf2;
    random = Metida.VarEffect(Metida.@covstr(factor|subject&factor), Metida.DIAG),
    repeated = Metida.VarEffect(Metida.@covstr(1|subject&factor), Metida.AR),
    )
    Metida.fit!(lmm)
    #@test Metida.m2logreml(lmm) ≈ 710.0962305879676 atol=1E-6

    @test  mean(Metida.rand(StableRNG(1234), lmm)) ≈ 50.12773055788859  # 50.435413902238096
    Metida.rand(lmm)
    Metida.rand(lmm, [4.54797, 2.82342, 1.05771, 0.576979])
    Metida.rand(lmm, [4.54797, 2.82342, 1.05771, 0.576979], [44.3, 5.3, 0.5, 0.29])
    v = zeros(nobs(lmm))
    @test mean(Metida.rand!(StableRNG(1234), v, lmm)) ≈  50.12773055788859 # 50.435413902238096
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
    @test_throws ErrorException Metida.reml_hessian(lmm)
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

    # Warn for non-unique levels for repeated effect within subject
    @test_warn "If UN structure used for repeated effect all levels should be unique within one subject, otherwise results can be meaningless!" Metida.LMM(@formula(log(lnpk)~sequence+period+treatment), dfrdsfda; repeated = Metida.VarEffect(Metida.@covstr(treatment|subject), Metida.UN))

    dfNaN = deepcopy(df0)
    dfNaN.var[1] = NaN
    lmm = Metida.LMM(@formula(var~sequence+period+formulation), dfNaN;
    random = Metida.VarEffect(Metida.@covstr(formulation|nosubj), Metida.DIAG),
    )
    @test_throws ErrorException Metida.fit!(lmm)

    dfNaN.var = string.(dfNaN.var)
    try 
        @test_warn  "Response variable not <: Union{Missing, AbstractFloat}, eltype: String" lmm = Metida.LMM(@formula(var~sequence+period+formulation), dfNaN; random = Metida.VarEffect(Metida.@covstr(formulation|nosubj), Metida.DIAG),)
    catch e
    end
    @test_throws  MethodError  Metida.LMM(@formula(var~sequence+period+formulation), dfNaN; random = Metida.VarEffect(Metida.@covstr(formulation|nosubj), Metida.DIAG),)


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
    @test Metida.dof_satter(lmm)[1] ≈ 87.00202572466458 atol=1E-2

###############################################################################
    lmm = Metida.LMM(@formula(r4 ~ f), spatdf;
    repeated = Metida.VarEffect(Metida.@covstr(x+y|1), Metida.SPEXPD),
    )
    Metida.fit!(lmm, maxthreads = 1)
    @test Metida.m2logreml(lmm) ≈ 1835.8648295317691 atol=1E-6
    @test Metida.dof_satter(lmm)[1] ≈ 6.147693839389808 atol=1E-2

    lmm = Metida.LMM(@formula(r3 ~ f), spatdf;
    repeated = Metida.VarEffect(Metida.@covstr(x+y|1), Metida.SPPOWD),
    )
    Metida.fit!(lmm, maxthreads = 1)
    @test Metida.m2logreml(lmm) ≈ 1899.3636384223198 atol=1E-6
    @test Metida.dof_satter(lmm)[1] ≈ 58.75904971406159 atol=1E-2

    lmm = Metida.LMM(@formula(r5 ~ f), spatdf;
    repeated = Metida.VarEffect(Metida.@covstr(x+y|1), Metida.SPGAUD),
    )
    Metida.fit!(lmm, maxthreads = 1)
    @test Metida.m2logreml(lmm) ≈ 1860.4865219180099 atol=1E-6
    @test Metida.dof_satter(lmm)[1] ≈ 119.7608528562911 atol=1E-2
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
