# Metida

using  Test, CSV, DataFrames, StatsModels, StatsBase, LinearAlgebra, CategoricalArrays, Random, StableRNGs

path    = dirname(@__FILE__)
include("testdata.jl")
include("references.jl")

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

    REML0  = ref("api_refmodel.reml")
    THETA0 = refv("api_refmodel.theta")

########################################################################
@testset "   1. Конструирование модели                              " begin
########################################################################

    # Все ожидаемые значения этой секции берутся из словаря эталонов, а каждая
    # проверка попадает в журнал: `@refcheck "ключ" выражение`.

    # --- @formula + VarEffect ------------------------------------------------
    lmm = refmodel()
    @test isa(lmm, Metida.LMM)
    @refcheck "api_refmodel.isfitted.unfitted" Metida.isfitted(lmm)
    @refcheck "api_refmodel.nobs"              Metida.nobs(lmm)
    @refcheck "api_refmodel.rankx"             Metida.rankx(lmm)
    @refcheck "api_refmodel.coefn"             Metida.coefn(lmm)
    @refcheck "api_refmodel.thetalength"       Metida.thetalength(lmm)
    @refcheck "api_refmodel.fixedeffn"         Metida.fixedeffn(lmm)
    @refcheck "api_refmodel.responsename"      Metida.responsename(lmm)
    @refcheck "api_refmodel.nblocks"           Metida.nblocks(lmm)
    @refcheck "api_refmodel.coefnames"         coefnames(lmm)
    @test_nowarn formula(lmm)
    @test_nowarn show(io, lmm)                      # неподогнанная модель печатается

    # --- @lmmformula ---------------------------------------------------------
    lmmf = Metida.LMM(Metida.@lmmformula(var ~ sequence + period + formulation,
        random = formulation | subject : Metida.DIAG), df0)
    Metida.fit!(lmmf)
    @refcheck "api_refmodel.reml"      Metida.m2logreml(lmmf)
    @refcheck "s1_lmmf.fixedeffn"      Metida.fixedeffn(lmmf)

    # без интерсепта
    lmmf0 = Metida.fit(Metida.LMM, Metida.@lmmformula(var ~ 0 + sequence + period + formulation,
        random = formulation | subject : Metida.DIAG), df0)
    @refcheck "s1_lmmf0.fixedeffn"      Metida.fixedeffn(lmmf0)
    @refcheck "s1_lmmf0.typeiii_nterms" length(Metida.typeiii(lmmf0).name)

    # функциональный терм в отклике
    lmmlog = Metida.fit(Metida.LMM, Metida.@lmmformula(log(var) ~ sequence + period + formulation,
        random = formulation | subject : Metida.DIAG), df0)
    @refcheck "s1_lmmlog.responsename" Metida.responsename(lmmlog)

    # --- fit(LMM, ...) как конструктор+подгонка ------------------------------
    lmmc = Metida.fit(Metida.LMM, @formula(var ~ sequence + period + formulation), df0;
        random = Metida.VarEffect(Metida.@covstr(formulation | subject), Metida.DIAG))
    @refcheck "api_refmodel.reml" Metida.m2logreml(lmmc)

    # --- Казуистика в спецификации субъекта ----------------------------------
    # subject = константный столбец, subject = литерал 1 и repeated|nosubj
    # должны давать один и тот же результат (один блок на все наблюдения).
    for subjspec in (Metida.@covstr(formulation | nosubj), Metida.@covstr(formulation | 1))
        l = Metida.LMM(@formula(var ~ sequence + period + formulation), df0;
            random = Metida.VarEffect(subjspec, Metida.DIAG))
        Metida.fit!(l)
        @refcheck "s1_l.m2logreml" Metida.m2logreml(l)
        @refcheck "s1_l.nblocks"   Metida.nblocks(l)
    end

    lrep = Metida.LMM(@formula(var ~ sequence + period + formulation), df0;
        repeated = Metida.VarEffect(Metida.@covstr(formulation | 1), Metida.DIAG))
    Metida.fit!(lrep)
    @refcheck "s1_lrep.m2logreml" Metida.m2logreml(lrep)

    lrep2 = Metida.LMM(@formula(var ~ sequence + period + formulation), df0;
        repeated = Metida.VarEffect(Metida.@covstr(formulation | nosubj)))
    Metida.fit!(lrep2)
    @refcheck "s1_lrep2.m2logreml" Metida.m2logreml(lrep2)

    # --- Кодирование в случайной части ---------------------------------------
    # По умолчанию для терма с интерсептом и для 0+ ставится FullDummyCoding
    li = Metida.LMM(@formula(var ~ 1), df0;
        random = Metida.VarEffect(Metida.@covstr(1 + formulation | subject)))
    @refcheck "s1_li.coding" string(nameof(typeof(li.covstr.random[1].coding[:formulation])))

    lz = Metida.LMM(@formula(var ~ 1), df0;
        random = Metida.VarEffect(Metida.@covstr(0 + formulation | subject)))
    @refcheck "s1_lz.coding" string(nameof(typeof(lz.covstr.random[1].coding[:formulation])))

    lc = Metida.LMM(@formula(var ~ 1), df0;
        random = Metida.VarEffect(Metida.@covstr(1 + formulation | subject),
                                  coding = Dict(:formulation => StatsModels.DummyCoding())))
    @refcheck "s1_lc.coding" string(nameof(typeof(lc.covstr.random[1].coding[:formulation])))

    # --- Единственный фиксированный эффект -----------------------------------
    onefe = Metida.LMM(@formula(var ~ 1), df0;
        random = Metida.VarEffect(Metida.@covstr(formulation | subject), Metida.DIAG))
    @refcheck "s1_onefe.coefnames" coefnames(onefe)
    @test_nowarn show(io, onefe)

    # --- Вектор repeated-эффектов --------------------------------------------
    lmv = Metida.LMM(@formula(var ~ sequence + period + formulation), df0;
        repeated = [Metida.VarEffect(Metida.@covstr(formulation | subject), Metida.DIAG),
                    Metida.VarEffect(Metida.@covstr(1 | subject), Metida.SI)])
    Metida.fit!(lmv)
    @refcheck "s1_lmv.isfitted"    Metida.isfitted(lmv)
    @refcheck "s1_lmv.thetalength" Metida.thetalength(lmv)
    @test_nowarn show(io, lmv)

end # 1



########################################################################
@testset "   2. Подгонка: значения REML и ML                        " begin
########################################################################
 
    # Ожидаемые значения - из словаря; соотношения между вычисленными
    # величинами идут через @refrel и тоже попадают в журнал.
 
    lmm = refmodel(); Metida.fit!(lmm)
 
    @refcheck "api_refmodel.isfitted.fitted" Metida.isfitted(lmm)
    @refcheck "api_refmodel.reml" Metida.m2logreml(lmm)
    @refcheck "api_refmodel.reml" Metida.m2logreml(lmm, Metida.theta(lmm)) "θ передана явно: то же значение"
    @refcheck "api_refmodel.reml" Metida.m2logreml(lmm, THETA0) "θ из словаря: то же значение"
    @refcheck "api_refmodel.m2logreml.at_theta_05_03_02" Metida.m2logreml(lmm, [0.5, 0.3, 0.2])
 
    # Соотношения между вариантами критерия
    @refrel "logREML = -0.5 · (-2logREML)" Metida.logreml(lmm) -0.5 * Metida.m2logreml(lmm) 1E-10
    @refcheck "api_refmodel.logreml" Metida.logreml(lmm)
    @refrel "logML = -0.5 · (-2logML)" Metida.logml(lmm) -0.5 * Metida.m2logml(lmm) 1E-10
 
    @refcheck "api_refmodel.m2logml"   Metida.m2logml(lmm)
    @refcheck "api_refmodel.m2logml.2" Metida.m2logml(lmm, coef(lmm))
    @refcheck "api_refmodel.m2logml.3" Metida.m2logml(lmm, coef(lmm), Metida.theta(lmm); maxthreads = 8)
 
    @refcheck "api_refmodel.theta" Metida.theta(lmm)
    @refrel "lmm.θ - алиас theta(lmm) через getproperty" lmm.θ Metida.theta(lmm)
    @refrel "lmm.β - алиас coef(lmm) через getproperty"  lmm.β coef(lmm)
 
    # Повторная подгонка идемпотентна
    Metida.fit!(lmm)
    @refcheck "api_refmodel.reml" Metida.m2logreml(lmm) "повторная подгонка идемпотентна"
    # Подгонка с заданным init
    Metida.fit!(lmm; init = THETA0)
    @refcheck "api_refmodel.reml" Metida.m2logreml(lmm) "подгонка со стартом init = θ оптимума"
    # refitinit: старт от предыдущего решения
    Metida.fit!(lmm; refitinit = true)
    @refcheck "api_refmodel.reml" Metida.m2logreml(lmm) "подгонка с refitinit = true"
    # hes = false не должен менять оценки
    lmmh = refmodel(); Metida.fit!(lmmh; hes = false)
    @refcheck "api_refmodel.reml" Metida.m2logreml(lmmh) "подгонка с hes = false: оценки те же"
 
end # 2


########################################################################
@testset "   3. Внутренние величины REML                            " begin
########################################################################
 
    lmm = refmodel(); Metida.fit!(lmm)
    dv  = Metida.LMMDataViews(lmm)
    n   = length(lmm.covstr.vcovblock)
 
    # Эталоны слагаемых - из словаря, разложения и свойства - через @refrel
    # и @refprop; в журнал попадает вся секция.
 
    reml, θs₂, remlθ₃, noerror =
        Metida.reml_sweep_β(lmm, dv, Metida.theta(lmm), Metida.coef(lmm))
    @refprop  "reml_sweep_β прошла без ошибок" noerror
    @refcheck "api_refmodel.reml" reml "значение из reml_sweep_β"
    @refcheck "api_refmodel.rvr"  remlθ₃ "r'V⁻¹r из reml_sweep_β"
 
    θ₁, θ₂, θ₃, noerror2 =
        Metida.core_sweep_β(lmm, dv, Metida.theta(lmm), Metida.coef(lmm), n)
    @refprop  "core_sweep_β прошла без ошибок" noerror2
    @refcheck "api_refmodel.logdetv" θ₁ "Σlog|V| из core_sweep_β"
    @refcheck "api_refmodel.rvr"     θ₃ "r'V⁻¹r из core_sweep_β: то же, что из reml_sweep_β"
 
    # θ₂ хранится верхним треугольником; Symmetric даёт X'V⁻¹X
    @refprop "θ₂ хранится верхним треугольником" istriu(θ₂)
    @refrel  "Symmetric(θ₂) - это X'V⁻¹X из reml_sweep_β" Symmetric(θ₂) θs₂ 1E-6
    logdetθ₂ = logdet(Symmetric(θ₂))
    @refcheck "api_refmodel.logdetxvx" logdetθ₂
 
    # Разложение критерия: REML = Σlog|V| + log|X'V⁻¹X| + r'V⁻¹r + (N-p)log2π
    creml = (length(lmm.data.yv) - lmm.rankx) * log(2π)
    @refcheck "api_refmodel.reml" (θ₁ + logdetθ₂ + θ₃ + creml) "разложение REML по слагаемым"
    # ML  = Σlog|V| + r'V⁻¹r + N log2π
    cml = length(lmm.data.yv) * log(2π)
    @refcheck "api_refmodel.m2logml" (θ₁ + θ₃ + cml) "разложение ML по слагаемым"
 
    # X'V⁻¹X должна быть положительно определена в точке оптимума
    @refprop "X'V⁻¹X положительно определена в оптимуме" isposdef(Symmetric(θ₂))
    # и согласована с vcov
    @refrel  "vcov = (X'V⁻¹X)⁻¹" inv(Symmetric(θ₂)) vcov(lmm) 1E-8
 
end # 3



########################################################################
@testset "   4. Интерфейс StatsAPI / StatsBase                      " begin
########################################################################
 
    lmm = refmodel(); Metida.fit!(lmm)
 
    # Размеры и критерии - из словаря; связи между величинами интерфейса
    # (stderror и vcov, AIC и -2logREML, интервал и оценка) - через @refrel
    # и @refprop.
 
    @refcheck "api_refmodel.isfitted.fitted" isfitted(lmm) "через StatsAPI"
    @refcheck "api_refmodel.islinear"        islinear(lmm)
    @refcheck "api_refmodel.nobs"            nobs(lmm) "через StatsAPI"
    @refcheck "api_refmodel.dof_residual"    dof_residual(lmm)
    @refcheck "api_refmodel.coefn"           length(coef(lmm))     "длина вектора коэффициентов"
    @refcheck "api_refmodel.coefn"           length(stderror(lmm)) "длина вектора ошибок"
    @refcheck "api_refmodel.coefn"           size(vcov(lmm), 1)    "порядок матрицы vcov"
    @refprop  "vcov квадратная" size(vcov(lmm), 1) == size(vcov(lmm), 2)
    @refcheck "api_refmodel.vcov"     vcov(lmm)[1, 1]
    @refcheck "api_refmodel.stderror" stderror(lmm)[1]
    # stderror - корень из диагонали vcov
    @refrel  "stderror = √diag(vcov)" stderror(lmm) sqrt.(diag(vcov(lmm))) 1E-10
    @refprop "vcov симметрична" issymmetric(Matrix(vcov(lmm)))
 
    @refrel  "размер X = (nobs, coefn)" size(modelmatrix(lmm)) (nobs(lmm), Metida.coefn(lmm))
    @refrel  "длина X = nobs · coefn" length(modelmatrix(lmm)) nobs(lmm) * Metida.coefn(lmm)
    @refprop "отклик - вектор" (response(lmm) isa Vector)
    @refcheck "api_refmodel.nobs"  length(response(lmm)) "длина вектора отклика"
    @refcheck "api_refmodel.coefn" size(crossmodelmatrix(lmm), 1) "порядок X'X"
 
    @refcheck "api_refmodel.dof"  dof(lmm)
    @refcheck "api_refmodel.aic"  aic(lmm)
    @refcheck "api_refmodel.bic"  bic(lmm)
    @refcheck "api_refmodel.aicc" aicc(lmm)
    @refcheck "api_refmodel.caic" Metida.caic(lmm)
    # AIC = -2logREML + 2d, где d = число параметров ковариации
    @refrel "AIC = -2logREML + 2·thetalength" aic(lmm) (Metida.m2logreml(lmm) + 2 * Metida.thetalength(lmm)) 1E-8
    @refrel "loglikelihood = logREML" loglikelihood(lmm) Metida.logreml(lmm) 1E-10
 
    ct = coeftable(lmm)
    @refcheck "api_refmodel.coefn" length(ct.rownms) "строк в coeftable столько же, сколько коэффициентов"
    @test_nowarn show(io, ct)
 
    # confint: три метода ddf, согласованность скалярной и векторной формы
    ci_all = Metida.confint(lmm)
    @refcheck "api_refmodel.coefn" length(ci_all) "интервалов столько же, сколько коэффициентов"
    @refcheck "api_refmodel.confint.satter_lower"   ci_all[end][1]
    @refrel   "confint(lmm, 6) - та же граница, что в векторной форме" Metida.confint(lmm, 6)[1] ci_all[end][1] 1E-10
    @refcheck "api_refmodel.confint.residual_lower" Metida.confint(lmm; ddf = :residual)[end][1]
    @test_nowarn Metida.confint(lmm; ddf = :contain)             # SPSS:7
    # нижняя граница строго меньше верхней, оценка внутри интервала
    for (i, c) in enumerate(ci_all)
        @refprop "нижняя граница меньше верхней, коэффициент $i" c[1] < c[2]
        @refprop "оценка внутри интервала, коэффициент $i" c[1] < coef(lmm)[i] < c[2]
    end
    # более широкий уровень даёт более широкий интервал
    c90 = Metida.confint(lmm; level = 0.90)
    c99 = Metida.confint(lmm; level = 0.99)
    @refprop "интервал уровня 99 % шире, чем 90 %" c99[1][2] - c99[1][1] > c90[1][2] - c90[1][1]
 
end # 4



########################################################################
@testset "   5. Матрицы модели                                      " begin
########################################################################
 
    lmm = refmodel(); Metida.fit!(lmm)
 
    # Матрицы сверяются по сумме элементов - это дешёвый отпечаток; всё
    # остальное о них - свойства и связи, и они тоже идут в журнал.
 
    G = Metida.gmatrix(lmm, 1)
    R = Metida.rmatrix(lmm, 1)
    V = Metida.vmatrix(lmm, 1)
    @refcheck "api_refmodel.gmatrix_sum" sum(G)
    @refcheck "api_refmodel.rmatrix_sum" sum(R)
    @refcheck "api_refmodel.vmatrix_sum" sum(V)
    @refprop "G положительно определена" Metida.gmatrixipd(lmm)
    @refprop "V симметрична" issymmetric(Matrix(V))
    @refprop "V положительно определена" isposdef(Symmetric(Matrix(V)))
    @refrel  "порядок V = размер блока" size(V, 1) length(lmm.covstr.vcovblock[1])
 
    # V(θ) через явный вектор параметров совпадает с V подогнанной модели
    @refrel "V(θ) при явной θ - та же матрица" Metida.vmatrix(Metida.theta(lmm), lmm, 1) V 1E-10
 
    # vmatrix! пишет в предоставленный буфер
    Vb = zeros(size(V))
    Metida.vmatrix!(Vb, Metida.theta(lmm), lmm, 1)
    @refrel "vmatrix! заполняет буфер той же матрицей" Symmetric(Vb) Symmetric(Matrix(V)) 1E-10
 
    # Все блоки собираются без ошибок и положительно определены
    for i in 1:Metida.nblocks(lmm)
        Vi = Metida.vmatrix(lmm, i)
        @refprop "V блока $i положительно определена" isposdef(Symmetric(Matrix(Vi)))
    end
 
    @refrel   "размер гессиана = (thetalength, thetalength)" size(Metida.reml_hessian(lmm)) (Metida.thetalength(lmm), Metida.thetalength(lmm))
    @refcheck "api_refmodel.reml_hessian_sum" sum(Metida.reml_hessian(lmm))
    @refprop  "гессиан симметричен до шестого знака" issymmetric(round.(Metida.reml_hessian(lmm); digits = 6))
 
    # Случайные эффекты по блокам
    @refcheck "api_refmodel.raneffn" Metida.raneffn(lmm)
    for i in 1:Metida.nblocks(lmm)
        @test_nowarn Metida.raneff(lmm, i)
    end
    # Для модели без случайной части raneff == nothing
    lmmr = Metida.LMM(@formula(var ~ sequence + period + formulation), df0;
        repeated = Metida.VarEffect(Metida.@covstr(formulation | subject), Metida.DIAG))
    Metida.fit!(lmmr)
    @refcheck "s5_lmmr.raneffn" Metida.raneffn(lmmr)
    @refprop  "без случайной части raneff даёт nothing" (Metida.raneff(lmmr, 1) === nothing)
 
end # 5



########################################################################
@testset "   6. Инференс: DF, контрасты, тип III                    " begin
########################################################################
 
    lmm = refmodel(); Metida.fit!(lmm)
    l3  = [0 0 1 0 0 0; 0 0 0 1 0 0; 0 0 0 0 1 0]      # контраст на period
 
    # Числа - из словаря; у четырёх из них источник внешний (SPSS), это видно
    # в журнале колонкой origin. Формы вызова и границы - @refrel и @refprop.
 
    # --- Satterthwaite: три формы вызова должны совпадать --------------------
    d_int = Metida.dof_satter(lmm, 6)
    d_vec = Metida.dof_satter(lmm, [0, 0, 0, 0, 0, 1])
    d_all = Metida.dof_satter(lmm)
    @refcheck "api_refmodel.dof_satter.coef6" d_int                    # SPSS:1
    @refrel   "вектор-контраст даёт то же DF, что номер коэффициента" d_vec d_int 1E-6
    @refrel   "векторная форма согласована с поэлементной" d_all[end] d_int 1E-6
    @refcheck "api_refmodel.coefn" length(d_all) "DF считается по каждому коэффициенту"
    @refprop  "все DF не меньше единицы" all(x -> x >= 1.0, d_all)
    @refprop  "все DF не больше остаточных" all(x -> x <= dof_residual(lmm), d_all)
 
    # --- Матричный контраст --------------------------------------------------
    @refcheck "api_refmodel.dof_satter.period" Metida.dof_satter(lmm, l3)   # SPSS:2
    @refrel   "lcontrast(lmm, 3) - тот же контраст, что l3" Metida.dof_satter(lmm, Metida.lcontrast(lmm, 3)) Metida.dof_satter(lmm, l3) 1E-6
    @refcheck "api_refmodel.fvalue.period" Metida.fvalue(lmm, l3)
 
    # lcontrast: размерность и ранг
    L3 = Metida.lcontrast(lmm, 3)
    @refcheck "api_refmodel.coefn" size(L3, 2) "ширина L - число коэффициентов"
    @refcheck "api_refmodel.lcontrast_rank.period" rank(L3)
    @test_nowarn Metida.lcontrast(lmm, 1)
    @test_nowarn Metida.lcontrast(lmm, 4)
    @test_throws ErrorException Metida.lcontrast(lmm, 99)
 
    # --- typeiii: три метода ddf --------------------------------------------
    t3 = Metida.typeiii(lmm)
    @refcheck "api_refmodel.typeiii_nterms" length(t3.name)
    @refprop  "все F неотрицательны" all(x -> x >= 0, t3.f)
    @refprop  "все p в [0, 1]" all(x -> 0 <= x <= 1, t3.pval)
    @refcheck "api_refmodel.typeiii_pval.formulation" t3.pval[4]       # SPSS:3
    @test_nowarn Metida.typeiii(lmm; ddf = :residual)
    @test_nowarn Metida.typeiii(lmm; ddf = :contain)                   # SPSS:8
    @test_nowarn show(io, t3)
 
    # typeiii и contrast на одном и том же контрасте дают одно p-значение
    ct = Metida.contrast(lmm, l3)
    @refrel   "typeiii и contrast на одном контрасте дают одно p" t3.pval[3] ct.pval[1] 1E-8
    @refcheck "api_refmodel.contrast_ndf.period" ct.ndf[1]
    @test_nowarn show(io, ct)
    @test_nowarn Metida.contrast(lmm, l3; name = "period", ddf = :residual)
    # прямое задание ddf числом
    ctd = Metida.contrast(lmm, l3; ddf = 14)
    @refcheck "api_refmodel.contrast_df.given" ctd.df[1]
 
    # --- estimate ------------------------------------------------------------
    e1 = Metida.estimate(lmm, [0, 0, 0, 0, 0, 1]; level = 0.9)
    @test_nowarn show(io, e1)
    eall = Metida.estimate(lmm; level = 0.9)
    @test_nowarn show(io, eall)
 
    # --- dof_contain ---------------------------------------------------------
    lmmc = Metida.LMM(@formula(var ~ period * formulation), df0;
        random = Metida.VarEffect(Metida.@covstr(formulation + sequence | nosubj), Metida.SI))
    Metida.fit!(lmmc)
    @refrel "θ, округлённая до печатной точности, даёт то же -2logREML" Metida.m2logreml(lmmc, [0.222283, 0.444566]) Metida.m2logreml(lmmc) 1E-6
    @refcheck "s6_lmmc.dof_contain.coef1" Metida.dof_contain(lmmc, 1)  # SPSS:9
    @refcheck "s6_lmmc.dof_contain.coef5" Metida.dof_contain(lmmc, 5)  # SPSS:9
 
    tt = Metida.typeiii(lmmc)
    @refcheck "s6_lmmc.typeiii_f.period"    tt.f[2]                    # SPSS:4
    @refcheck "s6_lmmc.typeiii_ndf.period"  tt.ndf[2]
    @refcheck "s6_lmmc.typeiii_df.period"   tt.df[2]                   # SPSS:4
    @refcheck "s6_lmmc.typeiii_pval.period" tt.pval[2]
    # Диагностика на случай платформенного расхождения (см. SPSS_VALIDATION.md):
    # df == 1.0 означает срабатывание заглушки df < 1 в dof_satter_, а не
    # законную оценку - при lclr = 3 значения из (1, 2] недостижимы.
    @refprop "DF не схлопнулся в заглушку df < 1" tt.df[2] > 2.0
 
end # 6


########################################################################
@testset "   7. Link-функции и стратегии первого шага               " begin
########################################################################
 
    mk() = Metida.LMM(@formula(var ~ sequence + period + formulation), df0;
        random = Metida.VarEffect(Metida.@covstr(1 + formulation | subject), Metida.CSH;
                                  coding = Dict(:formulation => StatsModels.DummyCoding())))
 
    # У каждой link-функции свой эталон: ключ несёт вариант подгонки, поэтому
    # в журнале видно, какое отображение проверялось.
 
    l = mk(); Metida.fit!(l; rholinkf = :sqsigm)
    @refcheck "s7_mk.m2logreml.rholinkf_sqsigm" Metida.m2logreml(l)
    @refcheck "s7_mk.dof_satter.rholinkf_sqsigm" Metida.dof_satter(l, [0, 0, 0, 0, 0, 1])
 
    l = mk(); Metida.fit!(l; rholinkf = :atan)
    @refcheck "s7_mk.m2logreml.rholinkf_atan" Metida.m2logreml(l)
 
    l = mk(); Metida.fit!(l; rholinkf = :psigm)
    @refcheck "s7_mk.m2logreml.rholinkf_psigm" Metida.m2logreml(l)
 
    l = mk(); Metida.fit!(l; varlinkf = :sq)
    @refcheck "s7_mk.m2logreml.varlinkf_sq" Metida.m2logreml(l)
 
    # --- aifirst -------------------------------------------------------------
    # Стратегия первого шага меняет путь, но не оптимум.
    lmm = refmodel()
    Metida.fit!(lmm; aifirst = :score)
    @refcheck "api_refmodel.reml" Metida.m2logreml(lmm) "первый шаг aifirst = :score"
    Metida.fit!(lmm; aifirst = :ai)
    @refcheck "api_refmodel.reml" Metida.m2logreml(lmm) "первый шаг aifirst = :ai"
    Metida.fit!(lmm; aifirst = :ai, init = THETA0)
    @refcheck "api_refmodel.reml" Metida.m2logreml(lmm) "aifirst = :ai со стартом из словаря"
    Metida.fit!(lmm; aifirst = :default)
    @refcheck "api_refmodel.reml" Metida.m2logreml(lmm) "первый шаг aifirst = :default"
 
    # --- альтернативные оптимизаторы -----------------------------------------
    lmm = refmodel(); Metida.fit!(lmm; optmethod = Metida.LBFGS_OM)
    @refcheck "api_refmodel.reml.alt_optimizer" Metida.m2logreml(lmm) "оптимизатор LBFGS"
    lmm = refmodel(); Metida.fit!(lmm; optmethod = Metida.BFGS_OM)
    @refcheck "api_refmodel.reml.alt_optimizer" Metida.m2logreml(lmm) "оптимизатор BFGS"
 
    # --- применение link-функций напрямую ------------------------------------
    # θ хранится в неограниченной шкале, `varlinkvecapply` переводит её в
    # шкалу параметров: exp для дисперсий, sigm для корреляций.
    @refcheck "s7_varlink.varlinkvecapply" Metida.varlinkvecapply([0.1, 0.1], [:var, :rho]; varlinkf = :exp, rholinkf = :sigm)
 
end # 7

########################################################################
@testset "   8. Потоки и воспроизводимость                          " begin
########################################################################
 
    # Однопоточный и многопоточный прогон должны совпадать в пределах
    # накопления ошибки суммирования по блокам. Эталон у обоих один и тот же -
    # это и есть содержание проверки; в журнале прогоны различает примечание.
    lmm1 = refmodel(); Metida.fit!(lmm1; maxthreads = 1)
    lmm4 = refmodel(); Metida.fit!(lmm4; maxthreads = 4)
    @refcheck "api_refmodel.reml"  Metida.m2logreml(lmm1) "один поток, maxthreads = 1"
    @refcheck "api_refmodel.reml"  Metida.m2logreml(lmm4) "четыре потока, maxthreads = 4"
    @refcheck "api_refmodel.theta" Metida.theta(lmm1)     "θ одного потока"
    @refcheck "api_refmodel.dof_satter.coef6" Metida.dof_satter(lmm1, 6) "DF одного потока"
    @refrel "число потоков не меняет -2logREML" Metida.m2logreml(lmm4) Metida.m2logreml(lmm1) 1E-6
    @refrel "число потоков не меняет θ" Metida.theta(lmm4) Metida.theta(lmm1) 1E-6
    @refrel "число потоков не меняет DF Сатертвейта" Metida.dof_satter(lmm4, 6) Metida.dof_satter(lmm1, 6) 1E-4
 
    # Модель с блоками разного размера (BE-подобная)
    be = Metida.LMM(@formula(var ~ sequence + period + formulation), df0;
        random   = Metida.VarEffect(Metida.@covstr(formulation | subject), Metida.CSH),
        repeated = Metida.VarEffect(Metida.@covstr(formulation | subject), Metida.DIAG))
    Metida.fit!(be; aifirst = :score)
    @refcheck "s8_be.m2logreml.aifirst_score" Metida.m2logreml(be)
    Metida.fit!(be; maxthreads = 1)
    @refcheck "s8_be.m2logreml" Metida.m2logreml(be) "блоки разного размера в один поток"
 
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

    # Веса из колонки: по имени и по символу - идентично
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

    # Неверная длина вектора весов - предупреждение, веса игнорируются
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
    @refcheck "s10_lmiss.m2logreml" Metida.m2logreml(lmiss)
    @refcheck "s10_lmiss.nobs" nobs(lmiss)
    @refrel   "исключена ровно одна строка с пропуском отклика" nobs(lmiss) (size(df0m, 1) - 1)
 
    # --- Неполные данные внутри субъекта (df1) -------------------------------
    linc = Metida.LMM(@formula(var ~ sequence + period + formulation), df1;
        random   = Metida.VarEffect(Metida.@covstr(formulation | subject), Metida.CSH),
        repeated = Metida.VarEffect(Metida.@covstr(formulation | subject), Metida.DIAG))
    Metida.fit!(linc; hes = false)
    @refcheck "s10_linc.m2logreml.hes_false" Metida.m2logreml(linc)
    @refcheck "s10_linc.dof_satter.coef6" Metida.dof_satter(linc, 6)
 
    # --- Целочисленный отклик + функциональный терм в случайной части --------
    lmmint = @refwarn "целочисленный отклик - предупреждение о типе" "Response variable not <: AbstractFloat" Metida.fit(Metida.LMM,
        Metida.@lmmformula(varint ~ formulation, random = 1 + var^2 | subject : Metida.SI), df0)
    Metida.fit!(lmmint)
    @refcheck "s10_lmmint.m2logreml" Metida.m2logreml(lmmint)
 
    # --- Неполный ранг X -----------------------------------------------------
    lrd = @refwarn "неполный ранг X - предупреждение конструктора" "Fixed-effect matrix not full-rank" Metida.LMM(
        @formula(lnpk ~ sequence + period + treatment + subject), dfrdsfda;
        random = Metida.VarEffect(Metida.@covstr(treatment | subject), Metida.DIAG))
    @refcheck "s10_lrd.rankx" Metida.rankx(lrd)
    @refcheck "s10_lrd.coefn" Metida.coefn(lrd)
    @refprop  "ранг X меньше числа коэффициентов" (Metida.rankx(lrd) < Metida.coefn(lrd))
    # API на неподогнанной модели не должен падать
    @refnowarn "coef на неподогнанной модели"     Metida.coef(lrd)
    @refnowarn "vcov на неподогнанной модели"     Metida.vcov(lrd)
    @refnowarn "stderror на неподогнанной модели" Metida.stderror(lrd)
    @refnowarn "подгонка модели неполного ранга"  Metida.fit!(lrd)
    @refnowarn "confint с DF Сатертвейта на модели неполного ранга" Metida.confint(lrd; level = 0.95, ddf = :satter)
    @refnowarn "lcontrast на модели неполного ранга" Metida.lcontrast(lrd, 5)
    @refnowarn "typeiii на модели неполного ранга"   Metida.typeiii(lrd)
    @refnowarn "печать модели неполного ранга"       show(io, lrd)
    # Отброшенные коэффициенты помечены NaN в vcov/stderror
    @refcheck "s10_lrd.stderror_nan" count(isnan, Metida.stderror(lrd))
    @refrel   "coef отдаёт все коэффициенты, включая отброшенный" length(Metida.coef(lrd)) Metida.coefn(lrd)
 
end # 10


########################################################################
@testset "  11. Логирование и отображение                           " begin
########################################################################
 
    lmm = refmodel(); Metida.fit!(lmm)
    @refcheck "api_refmodel.msgnum"      Metida.msgnum(lmm.log)
    @refcheck "api_refmodel.msgnum.info" Metida.msgnum(lmm.log, :INFO)
    @refprop  "лог отдаётся вектором" (Metida.getlog(lmm) isa Vector)
    @refnowarn "печать модели"  show(io, lmm)
    @refnowarn "печать лога"    show(io, Metida.getlog(lmm))
 
    # Лог не очищается при повторной подгонке - к нему добавляется запись
    n1 = Metida.msgnum(lmm.log)
    Metida.fit!(lmm)
    @refcheck "api_refmodel.msgnum.refit" Metida.msgnum(lmm.log)
    @refrel   "повторная подгонка добавляет к логу одну запись" Metida.msgnum(lmm.log) (n1 + 1)
 
    # Отображение всех типов ковариации, представленных в API
    for ct in (Metida.SI, Metida.DIAG, Metida.CS, Metida.CSH,
               Metida.AR, Metida.ARH, Metida.ARMA, Metida.TOEP, Metida.TOEPH, Metida.UN)
        @refnowarn "печать структуры $ct" show(io, ct)
    end
    for ctf in (Metida.TOEPP(2), Metida.TOEPHP(2))
        @refnowarn "печать структуры $ctf" show(io, ctf)
    end
    @refnowarn "печать VarEffect" show(io, Metida.VarEffect(Metida.@covstr(formulation | subject), Metida.DIAG))
 
end # 11


########################################################################
@testset "  12. Обработка ошибок                                    " begin
########################################################################
 
    # У раздела нет ожидаемых значений - есть ожидаемое поведение, поэтому все
    # проверки здесь идут в журнал формой @refthrows.
    lmm = refmodel()
    # Инференс на неподогнанной модели
    @refthrows "typeiii требует подогнанной модели"     ErrorException Metida.typeiii(lmm)
    @refthrows "dof_satter требует подогнанной модели"  ErrorException Metida.dof_satter(lmm, 1)
    @refthrows "contrast требует подогнанной модели"    ErrorException Metida.contrast(lmm, [0 0 0 0 0 1])
 
    Metida.fit!(lmm)
    # Несогласованная размерность контраста
    @refthrows "contrast с контрастом неверной ширины"   ErrorException Metida.contrast(lmm, [0 0 1])
    @refthrows "lcontrast с нулевым номером фактора"     ErrorException Metida.lcontrast(lmm, 0)
    @refthrows "dof_satter с контрастом неверной ширины" ErrorException Metida.dof_satter(lmm, [0 0 1])
 
end # 12


########################################################################
@testset "  13. Симуляция и ресемплинг                              " begin
########################################################################
 
    lmm = refmodel(); Metida.fit!(lmm)
    rng = StableRNG(20240502)
 
    # rand по подогнанной модели: эталон у длины выборки один - число
    # наблюдений модели, а различают формы вызова примечания в журнале.
    y1 = Metida.rand(rng, lmm)
    @refcheck "api_refmodel.nobs" length(y1) "длина выборки rand(rng, lmm)"
    @refprop  "выборка конечна" all(isfinite, y1)
    y2 = Metida.rand(rng, lmm, Metida.theta(lmm))
    @refcheck "api_refmodel.nobs" length(y2) "длина выборки rand с явной θ"
    y3 = Metida.rand(rng, lmm, Metida.theta(lmm), Metida.coef(lmm))
    @refcheck "api_refmodel.nobs" length(y3) "длина выборки rand с явными θ и β"
    # rand! в предоставленный буфер
    buf = zeros(nobs(lmm))
    Metida.rand!(rng, buf, lmm)
    @refprop "rand! заполнил буфер конечными числами" all(isfinite, buf)
    # Воспроизводимость по seed
    @refrel "один seed - одна и та же выборка" Metida.rand(StableRNG(1), lmm) Metida.rand(StableRNG(1), lmm)
 
    # Параметрический бутстрап (малое n - только смоук)
    br = Metida.bootstrap(lmm; n = 20, verbose = false, rng = StableRNG(20240503))
    @refnowarn "печать результата бутстрапа" show(io, br)
    cb = Metida.confint(br; level = 0.95, method = :bp)
    @refcheck "api_refmodel.coefn" length(cb) "бутстрап даёт интервал на каждый коэффициент"
    @refprop  "у всех бутстрап-интервалов нижняя граница ниже верхней" all(c -> c[1] < c[2], cb)
    for m in (:bp, :rbp, :norm, :bcnorm)
        @refnowarn "бутстрап-интервал методом $m" Metida.confint(br, 1; method = m)
    end
    @refthrows "неизвестный метод интервала"  ErrorException Metida.confint(br, 1; method = :nonexistent)
    @refthrows "неизвестная метрика интервала" ErrorException Metida.confint(br, 1; metric = :nonexistent)
 
end # 13

########################################################################
@testset "  14. Предельные средние: lsmeans и emmeans               " begin
########################################################################

    lmm  = refmodel(); Metida.fit!(lmm)
    lmmc = Metida.LMM(@formula(var ~ period * formulation), df0;
        random = Metida.VarEffect(Metida.@covstr(formulation + sequence | nosubj), Metida.SI))
    Metida.fit!(lmmc)

    ls = Metida.lsmeans(lmm, :formulation)
    @refcheck "api_refmodel.lsmean.formulation_1"    ls.means.estimate[1]    # ref SPSS
    @refcheck "api_refmodel.lsmean_se.formulation_1" ls.means.se[1]          # ref SPSS
    @refcheck "api_refmodel.lsmean_df.formulation_1" ls.means.df[1]          # ref SPSS
    @refprop  "строк столько, сколько уровней фактора" length(ls.means.name) == 2
    @refprop  "имя строки называет фактор и уровень" ls.means.name == ["formulation: 1", "formulation: 2"]
    @refnowarn "таблица предельных средних печатается" show(io, ls)

    # То же через estimate по L: lsmeans не добавляет численности, только L
    @refrel "lsmeans совпадает с estimate по тому же L" Metida.estimate(lmm, Metida.lsml(lmm, Dict(:formulation => "1"))).estimate[1] ls.means.estimate[1] 1E-12

    # Единственный смысл равных весов: среднее принадлежит модели, а не
    # частотам ячеек, и потому не зависит от кодировки контрастов.
    lmmh = Metida.LMM(@formula(var ~ sequence + period + formulation), df0;
        random = Metida.VarEffect(Metida.@covstr(formulation | subject), Metida.DIAG),
        contrasts = Dict(:sequence    => StatsModels.HelmertCoding(),
                         :period      => StatsModels.HelmertCoding(),
                         :formulation => StatsModels.HelmertCoding()))
    Metida.fit!(lmmh)
    lsh = Metida.lsmeans(lmmh, :period)
    lsd = Metida.lsmeans(lmm,  :period)
    @refrel  "предельные средние не зависят от кодировки" maximum(abs.(lsh.means.estimate .- lsd.means.estimate)) 0.0 1E-10
    @refrel  "их стандартные ошибки тоже" maximum(abs.(lsh.means.se .- lsd.means.se)) 0.0 1E-10
    @refrel  "и их DF" maximum(abs.(lsh.means.df .- lsd.means.df)) 0.0 1E-6
    @refprop "emmeans - то же самое" Metida.emmeans(lmm, :period).means.estimate == lsd.means.estimate

    # Комбинация уровней: последний названный фактор меняется быстрее
    lsc = Metida.lsmeans(lmm, [:sequence, :period, :formulation])
    @refprop "строк столько, сколько комбинаций уровней" length(lsc.means.name) == 16
    @refprop "последний названный фактор меняется быстрее" lsc.levels[1] == ["1", "1", "1"] && lsc.levels[2] == ["1", "1", "2"]


    @refprop "все ячейки плана оценимы" all(lsc.estimable)
    @refprop "у восьми ячеек нет наблюдений" count(iszero, lsc.n) == 8
    @refprop "у оценимых ячеек есть конечное среднее" all(isfinite, lsc.means.estimate)
    @refprop "число наблюдений в сумме равно размеру выборки" sum(lsc.n) == nrow(df0)

    # Попарные сравнения
    lsp = Metida.lsmeans(lmm, :formulation; pairwise = true)
    @refprop "пар столько, сколько сочетаний по две" length(lsp.pairs.name) == 1
    @refrel  "разность средних равна разности оценок" lsp.pairs.estimate[1] (lsp.means.estimate[1] - lsp.means.estimate[2]) 1E-12
    @refprop "без pairwise пар нет" isnothing(ls.pairs)
    @refnowarn "таблица с попарными сравнениями печатается" show(io, lsp)

    # Ковариата держится в своём выборочном среднем, и это можно переопределить
    lmmv = Metida.LMM(@formula(var ~ formulation * var2), df0;
        random = Metida.VarEffect(Metida.@covstr(formulation | subject), Metida.DIAG))
    Metida.fit!(lmmv)
    @refrel  "умолчание ковариаты - её выборочное среднее" Metida.lsmeans(lmmv, :formulation).means.estimate[1] Metida.lsmeans(lmmv, :formulation; at = Dict(:var2 => mean(df0.var2))).means.estimate[1] 1E-12
    @refprop "другое значение ковариаты даёт другое среднее" Metida.lsmeans(lmmv, :formulation; at = Dict(:var2 => 5.0)).means.estimate[1] != Metida.lsmeans(lmmv, :formulation).means.estimate[1]

    @refthrows "фактора нет в фиксированной части"   ArgumentError Metida.lsmeans(lmm, :nosuchfactor)
    @refthrows "фактор назван дважды"                ArgumentError Metida.lsmeans(lmm, [:period, :period])
    @refthrows "список факторов пуст"                ArgumentError Metida.lsmeans(lmm, Symbol[])
    @refthrows "ковариата названа фактором"          ArgumentError Metida.lsmeans(lmmv, :var2)
    @refthrows "at назван не для ковариаты"          ArgumentError Metida.lsmeans(lmmv, :formulation; at = Dict(:formulation => 1.0))
    @refthrows "взвешивание, кроме равного, не реализовано" ArgumentError Metida.lsmeans(lmm, :period; weights = :proportional)

    lmmi = Metida.LMM(@formula(var ~ 0 + period & formulation), df0;
        random = Metida.VarEffect(Metida.@covstr(formulation | subject), Metida.DIAG))
    Metida.fit!(lmmi)
    lsi = Metida.lsmeans(lmmi, :formulation)
    @refprop "фактор только внутри взаимодействия: число наблюдений неизвестно" all(==(-1), lsi.n)
    @refprop "при этом строки оценимы и среднее конечно" all(lsi.estimable) && all(isfinite, lsi.means.estimate)
    @refrel  "а среднее то же, что у модели с главными эффектами" lsi.means.estimate[1] Metida.lsmeans(lmmc, :formulation).means.estimate[1] 1E-10


    lmmr = Metida.LMM(@formula(var ~ 1 + period & formulation), df0;
        random = Metida.VarEffect(Metida.@covstr(formulation | subject), Metida.DIAG))
    Metida.fit!(lmmr)
    Lr = Metida.lsml(lmmr, Dict(:formulation => "1"))
    Xr = Metida.modelmatrix(lmmr)
    @refprop "ранг 8 из 9: один столбец выброшен" Metida.rankx(lmmr) == 8 && Metida.coefn(lmmr) == 9
    @refprop "L лежит в строчном пространстве X" norm(Lr' .- Lr' * (pinv(Xr) * Xr)) < 1E-8
    @refprop "но нагружен на выброшенный столбец" any(i -> abs(Lr[i]) > 1E-10, setdiff(1:Metida.coefn(lmmr), view(lmmr.pivotvec, 1:Metida.rankx(lmmr))))
    @refprop "и потому неоценим" !Metida.lsmestimable(lmmr, Lr, pinv(Xr) * Xr)


    lsr = Metida.lsmeans(lmmr, :formulation)
    i1  = findfirst(==(["1"]), lsr.levels)
    i2  = findfirst(==(["2"]), lsr.levels)
    @refprop "неоценима ровно одна строка из двух" count(lsr.estimable) == 1
    @refprop "у неоценимой строки NaN" !lsr.estimable[i1] && isnan(lsr.means.estimate[i1])
    @refprop "у оценимой есть и среднее, и ошибка" lsr.estimable[i2] && isfinite(lsr.means.estimate[i2]) && isfinite(lsr.means.se[i2])


    lmmfi = Metida.LMM(@formula(var ~ formulation & log(var2)), df0;
        random = Metida.VarEffect(Metida.@covstr(formulation | subject), Metida.DIAG))
    Metida.fit!(lmmfi)
    ficomp = Metida.lsmcomps(lmmfi.f.rhs.terms[end])
    fimean = mean(log.(df0.var2))
    fil    = try Metida.lsml(lmmfi, Dict(:formulation => "1")) catch e; e end
    @refprop "взаимодействие раскладывается на фактор и ещё одну компоненту" length(ficomp) == 2 && count(c -> isa(c, CategoricalTerm), ficomp) == 1
    @refprop "функциональный терм внутри взаимодействия: отказ либо mean(log(var2))" fil isa ArgumentError || (fil isa AbstractVector && count(x -> isapprox(x, fimean; rtol = 1E-12), fil) == 1)

end # 14

end # Publick API basic tests

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
    @test Metida.dof_satter(lmm)[2] ≈ 21.944281700360293 atol=1E-5
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
    @test Metida.dof_satter(lmm)[2] ≈ 22.31337200822804 atol=1E-5
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
    @test Metida.dof_satter(lmm)[2] ≈ 165.99999999999005 atol=1E-5
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
    @test Metida.dof_satter(lmm, [0, 1]) ≈ 23.093021655996232 atol=1E-5

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

    #@test collect(Metida.confint(lmm)[6]) ≈  [0.053789444388152474, 0.23713911100102136] atol=1E-6 # v0.17.2
    @test collect(Metida.confint(lmm)[6]) ≈  [0.05283150611443774, 0.23809704927585365] atol=1E-6
    anovatable = Metida.typeiii(lmm)
    # @test anovatable.pval ≈ [3.087934998046721e-63, 0.9176105002577626, 0.6522549061162943, 0.002010933915677479] atol=1E-4 # v0.17.2
    @test anovatable.pval ≈ [3.893289922345628e-63, 0.9176105366855528, 0.6524133797823746, 0.002503244153692245] atol=1E-4

    est = Metida.estimate(lmm, [0,0,0,0,0,1]; level = 0.9)
    @test est.t[1] ≈ 3.12818 atol=1E-4
   # @test est.pval[1] ≈ 0.0020 atol=1E-4 # v0.17.2
    @test est.pval[1] ≈ 0.002503244153692246 atol=1E-4
   # @test est.cil[1] ≈ 0.06863 atol=1E-4 # v0.17.2
    @test est.cil[1] ≈ 0.06802149627020912 atol=1E-4 
   # @test est.ciu[1] ≈ 0.2223 atol=1E-4  # v0.17.2
    @test est.ciu[1] ≈ 0.22290705912008227 atol=1E-4

    lmm = Metida.LMM(@formula(lnpk~0+sequence+period+treatment), dfrdsfda;
    random = Metida.VarEffect(Metida.@covstr(treatment|subject), Metida.CSH),
    repeated = Metida.VarEffect(Metida.@covstr(treatment|subject), Metida.DIAG),
    )
    Metida.fit!(lmm)
    anovatable = Metida.typeiii(lmm)
    #@test anovatable.pval ≈ [0.9176105002855397, 0.6522549061174356, 0.0020109339157131302] atol=1E-4 # v0.17.2
    @test anovatable.pval ≈ [0.9176105366858174, 0.6524133797977861, 0.0025032441543821602] atol=1E-4
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


include("test_acov_band_block.jl")

include("test_rtoe.jl")