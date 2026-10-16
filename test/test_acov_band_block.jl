########################################################################
# Модификаторы ковариации: ACOV (семантика action, SPPOW) и BAND(k),
# плюс предупреждение о граничной оценке ρ̂ (`rhotol`).
########################################################################
@testset "  Modifiers: ACOV / BAND / rho boundary                    " begin

    io = IOBuffer()

    # Эталоны раздела (ftdf3, блок 7x7, repeated по r1 внутри subject).
    ACOV_AR_DIAG = 821.4020237419716   # DIAG + ACOV(AR)  == ARH
    ACOV_CS_DIAG = 862.1310365008385   # DIAG + ACOV(CS)  == CSH
    CSH_PLAIN    = 862.1310365008401   # CSH
    CSH_BAND1    = 963.0584566224911   # CSH + BAND(1)    == TOEPHP(2)

    # Конструкторы моделей раздела.
    acov(base, mod; data = ftdf3) = Metida.LMM(@formula(response ~ 1), data;
        repeated = [Metida.VarEffect(Metida.@covstr(r1 | subject), base),
                    Metida.VarEffect(Metida.@covstr(1 | subject),  mod)])
    plain(base) = Metida.LMM(@formula(response ~ 1), ftdf3;
        repeated = [Metida.VarEffect(Metida.@covstr(r1 | subject), base)])
    fitq(lmm) = Metida.fit!(lmm; io = io)

########################################################################
@testset "   1. ACOV: семантика `action`                            " begin
########################################################################

    # `action` по документации срабатывает ТОЛЬКО на уже ненулевой ячейке.
    # На диагональной базе ненулевых ковариаций нет, поэтому все шесть
    # значений обязаны давать одну и ту же модель. До исправления
    # action = 2 и action = 5 зануляли новую ковариацию и сворачивали
    # модель в чистый DIAG (-2logREML 1086.795 вместо 821.402), оставляя
    # ρ в θ неидентифицируемым параметром.
    # Попутно: предупреждение о вакуумном `action` выдается ОДИН раз при
    # конструировании, а не на каждую ячейку на каждой итерации
    # оптимизатора (раньше это были тысячи сообщений за один fit!).
    for a = 0:5
        lmm = if a == 0
            @test_nowarn acov(Metida.DIAG, Metida.ACOV(Metida.AR; action = 0))
        else
            @test_logs (:warn,) match_mode = :any acov(Metida.DIAG, Metida.ACOV(Metida.AR; action = a))
        end
        fitq(lmm)
        @test Metida.m2logreml(lmm) ≈ ACOV_AR_DIAG atol = 1E-6
        @test Metida.thetalength(lmm) == 4
    end

    # На недиагональной базе ячейки ненулевые, и значения action обязаны
    # различаться: 0 - прибавить, 1 - заменить, 2 - сохранить имеющееся.
    l0 = acov(Metida.CS, Metida.ACOV(Metida.AR; action = 0)); fitq(l0)
    l1 = acov(Metida.CS, Metida.ACOV(Metida.AR; action = 1)); fitq(l1)
    l2 = acov(Metida.CS, Metida.ACOV(Metida.AR; action = 2)); fitq(l2)
    @test Metida.m2logreml(l0) ≈ 834.5529772600421 atol = 1E-6
    @test Metida.m2logreml(l1) ≈ 833.8867447431483 atol = 1E-6
    @test Metida.m2logreml(l2) ≈ 876.6510332438622 atol = 1E-6
    @test !isapprox(Metida.m2logreml(l0), Metida.m2logreml(l1), atol = 1E-6)
    @test !isapprox(Metida.m2logreml(l0), Metida.m2logreml(l2), atol = 1E-6)

    # action = 1 на базе CS должен совпасть с SI + ACOV(AR): «заменить»
    # стирает CS-ковариацию и оставляет только AR-приращение.
    lsi = Metida.LMM(@formula(response ~ 1), ftdf3;
        repeated = [Metida.VarEffect(Metida.@covstr(1 | subject), Metida.SI),
                    Metida.VarEffect(Metida.@covstr(1 | subject), Metida.ACOV(Metida.AR))])
    fitq(lsi)
    @test Metida.m2logreml(l1) ≈ Metida.m2logreml(lsi) atol = 1E-6

    # Осмысленный случай: action 3..5 на недиагональной базе - предупреждение;
    # штатные 0/1/2 на ней молчат.
    @test_logs (:warn,) match_mode = :any acov(Metida.CS, Metida.ACOV(Metida.AR; action = 3))
    @test_logs (:warn,) match_mode = :any acov(Metida.CS, Metida.ACOV(Metida.AR; action = 4))
    @test_logs (:warn,) match_mode = :any acov(Metida.CS, Metida.ACOV(Metida.AR; action = 5))
    @test_nowarn acov(Metida.CS, Metida.ACOV(Metida.AR; action = 1))
    @test_nowarn acov(Metida.CS, Metida.ACOV(Metida.AR; action = 2))

    # action > 5 - ошибка, но только когда ячейка действительно ненулевая.
    @test_throws Exception fitq(acov(Metida.CS, Metida.ACOV(Metida.AR; action = 6)))
    lmm = @test_logs (:warn,) match_mode = :any acov(Metida.DIAG, Metida.ACOV(Metida.AR; action = 6))
    fitq(lmm)                                                     # ненулевых ячеек нет
    @test Metida.m2logreml(lmm) ≈ ACOV_AR_DIAG atol = 1E-6
end

########################################################################
@testset "   2. ACOV: тождества эквивалентности                     " begin
########################################################################

    # При одном и том же факторе группировки композиция воспроизводит
    # встроенную структуру. Расхождение - обычный допуск оптимизатора
    # (~1E-12), не разные модели.
    la = acov(Metida.DIAG, Metida.ACOV(Metida.AR)); fitq(la)
    lb = plain(Metida.ARH);                         fitq(lb)
    @test Metida.m2logreml(la) ≈ Metida.m2logreml(lb)
    @test Metida.thetalength(la) == Metida.thetalength(lb)

    lc = acov(Metida.DIAG, Metida.ACOV(Metida.CS)); fitq(lc)
    ld = plain(Metida.CSH);                         fitq(ld)
    @test Metida.m2logreml(lc) ≈ Metida.m2logreml(ld)
    @test Metida.thetalength(lc) == Metida.thetalength(ld)

    le = Metida.LMM(@formula(response ~ 1), ftdf3;
        repeated = [Metida.VarEffect(Metida.@covstr(1 | subject), Metida.SI),
                    Metida.VarEffect(Metida.@covstr(1 | subject), Metida.ACOV(Metida.AR))])
    lg = Metida.LMM(@formula(response ~ 1), ftdf3;
        repeated = [Metida.VarEffect(Metida.@covstr(1 | subject), Metida.AR)])
    fitq(le); fitq(lg)
    @test Metida.m2logreml(le) ≈ Metida.m2logreml(lg)

    # Композиция аддитивна, а не разделима: ячейки, различающиеся по ОБОИМ
    # факторам, не получают вклада ни от одного слоя и остаются нулями.
    # Для блока 8x8 (4 прибора x 2 дня) это 12 структурных нулей из 28.
    lun = Metida.LMM(@formula(resp ~ 0 + device), devday;
        repeated = [Metida.VarEffect(Metida.@covstr(device | subj & day), Metida.UN),
                    Metida.VarEffect(Metida.@covstr(1 | subj & device), Metida.ACOV(Metida.AR))])
    fitq(lun)
    rm8 = Metida.rmatrix(lun, 1)
    @test size(rm8) == (8, 8)
    @test count(!iszero, [rm8[m, n] for n = 1:8 for m = 1:n-1]) == 16
    @test Metida.m2logreml(lun) ≈ 903.0575888146 atol = 1E-6
end

########################################################################
@testset "   3. ACOV(SPPOW): лаг как расстояние                     " begin
########################################################################

    # У ACOV(AR) лаг - разность НОМЕРОВ СТРОК внутри блока, поэтому
    # перестановка строк данных меняет модель. ACOV(SPPOW) берет лагом
    # евклидово расстояние между строками rz, то есть фактический
    # интервал по координате, и к порядку строк безразличен.
    dn    = copy(ftdf3)
    dn.pn = parse.(Float64, dn.p)                 # p: "1".."7" -> числовая координата
    ds    = dn[shuffle(StableRNG(42), 1:size(dn, 1)), :]

    sppow(data) = Metida.LMM(@formula(response ~ 1), data;
        repeated = [Metida.VarEffect(Metida.@covstr(r1 | subject), Metida.DIAG),
                    Metida.VarEffect(Metida.@covstr(pn | subject), Metida.ACOV(Metida.SPPOW))])

    lsp  = sppow(dn);  fitq(lsp)
    lsps = sppow(ds);  fitq(lsps)
    lar  = acov(Metida.DIAG, Metida.ACOV(Metida.AR); data = dn);  fitq(lar)
    lars = acov(Metida.DIAG, Metida.ACOV(Metida.AR); data = ds);  fitq(lars)

    @test Metida.thetalength(lsp) == 4                       # ρ ровно один
    @test strip(lsp.covstr.rcnames[end]) == "ρ"              # имя параметра не заглушка
    # на равномерной сетке SPPOW воспроизводит AR
    @test Metida.m2logreml(lsp) ≈ ACOV_AR_DIAG atol = 1E-6
    @test Metida.m2logreml(lar) ≈ ACOV_AR_DIAG atol = 1E-6
    # SPPOW инвариантен к порядку строк, AR - нет
    @test Metida.m2logreml(lsps) ≈ Metida.m2logreml(lsp)  atol = 1E-6
    @test Metida.m2logreml(lars) ≈ 926.2148730800657      atol = 1E-6
    @test !isapprox(Metida.m2logreml(lars), Metida.m2logreml(lar), atol = 1E-6)

    @test_nowarn show(io, Metida.ACOV(Metida.SPPOW))
end

########################################################################
@testset "   4. BAND(k): маска по лагу                              " begin
########################################################################

    # BAND(k) зануляет каждую ячейку с внутриблочным лагом > k и не
    # добавляет ни одного параметра. Полоса не блочно-диагональна,
    # поэтому фактором группировки не выражается.
    lb1 = acov(Metida.CSH, Metida.BAND(1)); fitq(lb1)
    lb6 = acov(Metida.CSH, Metida.BAND(6)); fitq(lb6)
    lcsh = plain(Metida.CSH);               fitq(lcsh)
    ltp  = plain(Metida.TOEPHP(2));         fitq(ltp)

    # ни одного нового параметра
    @test Metida.thetalength(lb1) == Metida.thetalength(lcsh) == 4
    @test Metida.dof(lb1) == Metida.dof(lcsh)

    # перекрестная проверка против уже существующей структуры
    @test Metida.m2logreml(lb1) ≈ CSH_BAND1 atol = 1E-6
    @test Metida.m2logreml(lb1) ≈ Metida.m2logreml(ltp) atol = 1E-6

    # k не меньше размера блока - полоса ничего не отсекает
    @test Metida.m2logreml(lb6) ≈ CSH_PLAIN atol = 1E-6
    @test Metida.m2logreml(lb6) ≈ Metida.m2logreml(lcsh) atol = 1E-6

    # структура маски в самой R-матрице
    lb2 = acov(Metida.CSH, Metida.BAND(2)); fitq(lb2)
    rm  = Metida.rmatrix(lb2, 1)
    s   = size(rm, 1)
    @test s == 7
    @test all(iszero(rm[m, n]) for n = 1:s for m = 1:n-1 if n - m >  2)
    @test all(!iszero(rm[m, n]) for n = 1:s for m = 1:n-1 if n - m <= 2)
    @test isposdef(rm)

    @test_nowarn show(io, Metida.BAND(3))
    @test_nowarn show(io, lb1)
end

########################################################################
@testset "   5. Модификаторы: ошибки конструирования                " begin
########################################################################

    # Модификатор не может быть первым повторным эффектом: блок еще
    # нулевой, sqrt(0*0)*ρ == 0, параметр неидентифицируем. Раньше это
    # было предупреждением, подгонка шла дальше и возвращала мусор при
    # isfitted == false.
    @test_throws ErrorException Metida.LMM(@formula(response ~ 1), ftdf3;
        repeated = Metida.VarEffect(Metida.@covstr(1 | subject), Metida.ACOV(Metida.AR)))
    @test_throws ErrorException Metida.LMM(@formula(response ~ 1), ftdf3;
        repeated = Metida.VarEffect(Metida.@covstr(1 | subject), Metida.BAND(1)))

    # Отрицательная ширина полосы.
    @test_throws ArgumentError Metida.BAND(-1)
    @test_nowarn Metida.BAND(0)

    # Внутренние типы, кроме AR / CS / SPPOW, по-прежнему отвергаются.
    @test_throws ErrorException Metida.LMM(@formula(response ~ 1), ftdf3;
        repeated = [Metida.VarEffect(Metida.@covstr(r1 | subject), Metida.DIAG),
                    Metida.VarEffect(Metida.@covstr(1 | subject), Metida.ACOV(Metida.ARH))])
end

########################################################################
@testset "   6. rhotol: граничная оценка ρ̂                          " begin
########################################################################

    # Link-функции отображают R на ОТКРЫТЫЙ интервал, поэтому |ρ̂| == 1
    # ровно не достигается и ни одна прежняя проверка не срабатывала.
    # Оценка на расстоянии < rhotol от ±1 означает, что правдоподобие
    # все еще растет на границе области: V численно вырождена в этом
    # направлении, гессиан плохо обусловлен, DF и SE недостоверны.
    bnd() = Metida.LMM(@formula(var ~ sequence + period + formulation), df0;
        random   = Metida.VarEffect(Metida.@covstr(formulation | subject), Metida.CSH),
        repeated = Metida.VarEffect(Metida.@covstr(formulation | subject), Metida.DIAG))

    lmm = bnd(); Metida.fit!(lmm; io = io)
    @test Metida.theta(lmm)[3] ≈ 1.0 atol = 1E-8          # ρ̂ на границе
    @test Metida.msgnum(lmm.log, :WARN) == 1
    @test Metida.msgnum(lmm.log) == 4
    @test occursin("boundary", lmm.log[end].msg)
    # проверка ничего не вычисляет - только читает result.theta
    @test Metida.m2logreml(lmm) ≈ 10.065238626765524 atol = 1E-6

    # rhotol = 0 отключает проверку (|ρ̂| < 1 всегда, строго)
    lmm = bnd(); Metida.fit!(lmm; io = io, rhotol = 0.0)
    @test Metida.msgnum(lmm.log, :WARN) == 0
    @test Metida.msgnum(lmm.log) == 3
    @test Metida.m2logreml(lmm) ≈ 10.065238626765524 atol = 1E-6

    # допуск соблюдается в обе стороны: внутренняя точка ρ̂ = 0.9199
    # молчит по умолчанию и срабатывает при широком rhotol
    lmm = acov(Metida.DIAG, Metida.ACOV(Metida.AR))
    Metida.fit!(lmm; io = io)
    @test Metida.theta(lmm)[end] ≈ 0.9199381881518123 atol = 1E-6
    @test Metida.msgnum(lmm.log, :WARN) == 0

    lmm = acov(Metida.DIAG, Metida.ACOV(Metida.AR))
    Metida.fit!(lmm; io = io, rhotol = 0.5)
    @test Metida.msgnum(lmm.log, :WARN) == 1
    @test Metida.m2logreml(lmm) ≈ ACOV_AR_DIAG atol = 1E-6

    # на структурах без корреляционных параметров проверка не срабатывает
    lmm = plain(Metida.DIAG); Metida.fit!(lmm; io = io, rhotol = 0.5)
    @test Metida.msgnum(lmm.log, :WARN) == 0
end

end
