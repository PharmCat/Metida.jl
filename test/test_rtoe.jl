########################################################################
# Новые структуры: TOEP и TOEPH как ПОВТОРНЫЕ эффекты, плюс BAND(k) на
# базах, не покрытых блоком модификаторов (там только CSH).
#
# Внешней сверки (SPSS) для этих структур пока нет, поэтому блок стоит на
# двух опорах, которые её не требуют:
#   * тождества с уже валидированными структурами там, где модели обязаны
#     совпадать по построению (вырождение Тёплица при t = 2, совпадение с
#     параметризованными вариантами при p = t, схлопывание BAND(1));
#   * инварианты, не зависящие от эталона: порядок вложенности по
#     гибкости, независимость от link-функции и от числа потоков,
#     положительная определённость, согласованность имён θ с `covstrparam`.
#
# Блок самодостаточен: использует только наборы из testdata.jl
# (`devday`, `df0`, `ftdf3`) и не меняет их.
########################################################################
@testset "  Structures: TOEP / TOEPH repeated, BAND bases            " begin

    io = IOBuffer()
    # Гессиан нужен только разделу 5 (Satterthwaite DF), а считается он здесь
    # дольше самой подгонки: 17.8 с против 0.009 с для TOEPH на devday. Все
    # разделы, которым нужны только -2logREML и θ, подгоняются без него.
    fitq(lmm)  = Metida.fit!(lmm; io = io, hes = false)
    fitqh(lmm) = Metida.fit!(lmm; io = io)

    # Каждая модель этого блока подгоняется РОВНО ОДИН раз: без мемоизации
    # разделы 5 и 10 переподгоняют одни и те же модели по два-три раза и
    # блок становится дороже всего остального test.jl вместе взятого.
    _MEMO = Dict{Any, Any}()
    memo(key, build; hes = false) =
        get!(() -> (l = build(); hes ? fitqh(l) : fitq(l); l), _MEMO, key)

    # Конструкторы: devday даёт две чистые группировки, в которых каждый
    # уровень встречается ровно один раз внутри субъекта - предусловие
    # уровневых структур (см. раздел 11).
    #   day|subj&device    -> блок 2x2, t = 2
    #   device|subj&day    -> блок 4x4, t = 4
    dd2(ct) = Metida.LMM(@formula(resp ~ 0 + device), devday;
        repeated = Metida.VarEffect(Metida.@covstr(day | subj & device), ct))
    dd4(ct) = Metida.LMM(@formula(resp ~ 0 + device), devday;
        repeated = Metida.VarEffect(Metida.@covstr(device | subj & day), ct))
    f2(ct) = memo((:dd2, ct), () -> dd2(ct))
    f4(ct) = memo((:dd4, ct), () -> dd4(ct))

########################################################################
@testset "   1. t = 2: Тёплиц вырождается в CS / CSH / UN           " begin
########################################################################

    # На двух уровнях структура Тёплица имеет ровно один лаг, поэтому
    # TOEP обязан совпасть с CS, а TOEPH - с CSH и, поскольку при t = 2
    # неструктурированная матрица тоже описывается двумя дисперсиями и
    # одной корреляцией, ещё и с UN. Это самая сильная проверка, какую
    # можно поставить без внешнего эталона: новая структура садится на
    # три уже валидированные.
    lt, lcs, ltp = f2(Metida.TOEP),  f2(Metida.CS),  f2(Metida.TOEPP(2))
    @test Metida.isfitted(lt)
    @test Metida.thetalength(lt) == Metida.thetalength(lcs) == 2
    @test Metida.m2logreml(lt) ≈ 1015.4115802326 atol = 1E-6
    @test Metida.m2logreml(lt) ≈ Metida.m2logreml(lcs) atol = 1E-8
    @test Metida.m2logreml(lt) ≈ Metida.m2logreml(ltp) atol = 1E-8

    lth, lcsh, lun, lthp = f2(Metida.TOEPH), f2(Metida.CSH), f2(Metida.UN), f2(Metida.TOEPHP(2))
    @test Metida.isfitted(lth)
    @test Metida.thetalength(lth) == Metida.thetalength(lcsh) == Metida.thetalength(lun) == 3
    @test Metida.m2logreml(lth) ≈ 1014.3605321044 atol = 1E-6
    @test Metida.m2logreml(lth) ≈ Metida.m2logreml(lcsh) atol = 1E-8
    @test Metida.m2logreml(lth) ≈ Metida.m2logreml(lun)  atol = 1E-8
    @test Metida.m2logreml(lth) ≈ Metida.m2logreml(lthp) atol = 1E-8

    # структура блока
    @test size(Metida.rmatrix(lt, 1))  == (2, 2)
    @test Metida.nblocks(lt) == 100
    @test isposdef(Metida.rmatrix(lth, 1))
end

########################################################################
@testset "   2. t = 4: совпадение с TOEPP(t) / TOEPHP(t)            " begin
########################################################################

    # При размере блока, равном числу уровней, и уровнях в их порядке
    # лаг по индексу уровня и лаг по позиции строки - одно и то же,
    # поэтому новые структуры обязаны совпасть с параметризованными при
    # p = t. Расходятся они только на неполных данных (раздел 8).
    lt,  ltp  = f4(Metida.TOEP),  f4(Metida.TOEPP(4))
    lth, lthp = f4(Metida.TOEPH), f4(Metida.TOEPHP(4))
    @test Metida.thetalength(lt)  == 4      # 1 дисперсия + 3 лага
    @test Metida.thetalength(lth) == 7      # 4 дисперсии + 3 лага
    @test Metida.m2logreml(lt)  ≈ 1032.5129102725 atol = 1E-6
    @test Metida.m2logreml(lth) ≈ 989.0021785107  atol = 1E-6
    @test Metida.m2logreml(lt)  ≈ Metida.m2logreml(ltp)  atol = 1E-8
    @test Metida.m2logreml(lth) ≈ Metida.m2logreml(lthp) atol = 1E-8
    @test Metida.dof(lt)  == 8
    @test Metida.dof(lth) == 11

    # Порядок вложенности по гибкости. UN содержит TOEPH как ограничение
    # (Тёплицева структура корреляций), TOEPH содержит TOEP (общая
    # дисперсия вместо поуровневых).
    lun  = f4(Metida.UN);  lcsh = f4(Metida.CSH)
    larh = f4(Metida.ARH); ldg  = f4(Metida.DIAG)
    @test Metida.m2logreml(lun) <= Metida.m2logreml(lth) <= Metida.m2logreml(lt)
    @test Metida.m2logreml(lcsh) ≈ 999.49712587    atol = 1E-6
    @test Metida.m2logreml(larh) ≈ 1006.3276490142 atol = 1E-6
    @test Metida.m2logreml(ldg)  ≈ 1008.4959849465 atol = 1E-6
    # на этих данных TOEPH выигрывает по AIC у всех перечисленных
    @test Metida.aic(lth) < min(Metida.aic(lun), Metida.aic(lcsh),
                                Metida.aic(larh), Metida.aic(ldg), Metida.aic(lt))
end

########################################################################
@testset "   3. второй набор данных: df0, period|subject            " begin
########################################################################

    # Независимое подтверждение тождеств на других данных и другой
    # фиксированной части.
    mkp(ct) = memo((:df0p, ct), () -> Metida.LMM(@formula(var ~ sequence + formulation), df0;
        repeated = Metida.VarEffect(Metida.@covstr(period | subject), ct)))
    lt,  ltp  = mkp(Metida.TOEP),  mkp(Metida.TOEPP(4))
    lth, lthp = mkp(Metida.TOEPH), mkp(Metida.TOEPHP(4))
    lun       = mkp(Metida.UN)
    @test Metida.m2logreml(lt)  ≈ 4.3163541053 atol = 1E-6
    @test Metida.m2logreml(lth) ≈ 3.0086147551 atol = 1E-6
    @test Metida.m2logreml(lt)  ≈ Metida.m2logreml(ltp)  atol = 1E-8
    @test Metida.m2logreml(lth) ≈ Metida.m2logreml(lthp) atol = 1E-8
    @test Metida.m2logreml(lun) <= Metida.m2logreml(lth) <= Metida.m2logreml(lt)
    # все корреляционные параметры строго внутри интервала
    ct = lt.covstr.ct
    @test all(i -> abs(Metida.theta(lt)[i]) < 1 - 1E-8, findall(==(:rho), ct))
end

########################################################################
@testset "   4. имена и типы параметров                             " begin
########################################################################

    # Имена не должны быть нумерованной заглушкой generic-метода, а их
    # число и порядок обязаны соответствовать covstrparam: (1, t-1) для
    # TOEP и (t, t-1) для TOEPH.
    lt, lth = f4(Metida.TOEP), f4(Metida.TOEPH)
    @test strip.(lt.covstr.rcnames)  == ["σ²", "ρ band 1", "ρ band 2", "ρ band 3"]
    @test lt.covstr.ct  == [:var, :rho, :rho, :rho]
    @test strip.(lth.covstr.rcnames) == ["σ² device: A", "σ² device: B",
                                         "σ² device: C", "σ² device: D",
                                         "ρ band 1", "ρ band 2", "ρ band 3"]
    @test lth.covstr.ct == [:var, :var, :var, :var, :rho, :rho, :rho]
    @test length(lt.covstr.rcnames)  == Metida.thetalength(lt)
    @test length(lth.covstr.rcnames) == Metida.thetalength(lth)
end

########################################################################
@testset "   5. инференс поверх новых структур                      " begin
########################################################################

    # собственные подгонки С гессианом, не из общего кэша
    f4h(ct) = memo((:dd4h, ct), () -> dd4(ct); hes = true)
    for ct in (Metida.TOEP, Metida.TOEPH)
        l = f4h(ct)
        d = Metida.dof_satter(l)
        @test length(d) == Metida.coefn(l)
        @test all(isfinite, d) && all(>(0), d)
        t = Metida.typeiii(l)
        @test all(isfinite, t.f) && all(isfinite, t.df) && all(isfinite, t.pval)
        c = Metida.confint(l)
        @test all(x -> all(isfinite, x), c)
        @test all(x -> x[1] < x[2], c)
        @test isposdef(Metida.vmatrix(Metida.theta(l), l, 1))
        @test all(isfinite, Metida.stderror(l))
        @test_nowarn Metida.confint(l; ddf = :residual)
    end
    # значения DF, чтобы численная регрессия падала, а не проходила молча
    # (модели уже в кэше, повторной подгонки нет)
    @test Metida.dof_satter(f4h(Metida.TOEP))[1]  ≈ 182.2052 atol = 1E-2
    @test Metida.dof_satter(f4h(Metida.TOEPH))[1] ≈ 48.3782  atol = 1E-2
end

########################################################################
@testset "   6. независимость от link-функции                       " begin
########################################################################

    # Репараметризация не должна менять аргмаксимум: все варианты link
    # обязаны привести в одну точку. Это и проверка того, что θ попадает
    # в нужные ветви varlinkvecapply по вектору ct.
    for (ct, ref) in ((Metida.TOEP, 1032.5129102725), (Metida.TOEPH, 989.0021785107))
        for kw in ((varlinkf = :identity,), (rholinkf = :atan,), (rholinkf = :psigm,))
            l = dd4(ct)
            Metida.fit!(l; io = io, hes = false, kw...)
            @test Metida.isfitted(l)
            @test Metida.m2logreml(l) ≈ ref atol = 1E-5
        end
    end
end

########################################################################
@testset "   7. потоки и воспроизводимость                          " begin
########################################################################

    for (ct, ref) in ((Metida.TOEP, 1032.5129102725), (Metida.TOEPH, 989.0021785107))
        for mt in (1, 16)
            l = dd4(ct)
            Metida.fit!(l; io = io, hes = false, maxthreads = mt)
            @test Metida.m2logreml(l) ≈ ref atol = 1E-8
        end
    end
end

########################################################################
@testset "   8. неполные данные: лаг следует за уровнем             " begin
########################################################################

    # Здесь две конвенции расходятся, и это единственная причина, по
    # которой уровневые структуры нужны рядом с параметризованными.
    # Убираем одно занятие: блок держит уровни A, C, D из A, B, C, D.
    # По уровням лаг (A,C) = 2 и (A,D) = 3; по позициям строк - 1 и 2.
    dd = devday[setdiff(1:size(devday, 1),
            findfirst(r -> r.subj == "1" && r.day == "d1" && r.device == "B",
                      eachrow(devday))), :]
    mki(ct) = memo((:ddinc, ct), () -> Metida.LMM(@formula(resp ~ 0 + device), dd;
        repeated = Metida.VarEffect(Metida.@covstr(device | subj & day), ct)))
    li,  lip  = mki(Metida.TOEP),  mki(Metida.TOEPP(4))
    lih, lihp = mki(Metida.TOEPH), mki(Metida.TOEPHP(4))
    lun       = mki(Metida.UN)

    @test Metida.thetalength(li)  == Metida.thetalength(lip)
    @test Metida.thetalength(lih) == Metida.thetalength(lihp)
    @test Metida.m2logreml(li)   ≈ 1027.8835756294 atol = 1E-5
    @test Metida.m2logreml(lip)  ≈ 1028.0716817182 atol = 1E-5
    @test Metida.m2logreml(lih)  ≈ 984.6116453274  atol = 1E-5
    @test Metida.m2logreml(lihp) ≈ 984.9675695641  atol = 1E-5
    # при равном числе параметров модели РАЗНЫЕ
    @test !isapprox(Metida.m2logreml(li),  Metida.m2logreml(lip),  atol = 1E-4)
    @test !isapprox(Metida.m2logreml(lih), Metida.m2logreml(lihp), atol = 1E-4)
    # вложенность сохраняется и на неполных данных
    @test Metida.m2logreml(lun) <= Metida.m2logreml(lih) <= Metida.m2logreml(li)

    # сами лаги, считанные из усечённого блока 3x3
    θ  = Metida.theta(li); σ² = θ[1]^2
    bi = findfirst(b -> size(Metida.rmatrix(li, b), 1) == 3, 1:Metida.nblocks(li))
    @test !isnothing(bi)
    r  = Metida.rmatrix(li, bi)
    @test r[1, 2] / σ² ≈ θ[3] atol = 1E-6    # (A, C): уровни 1 и 3 -> лаг 2
    @test r[1, 3] / σ² ≈ θ[4] atol = 1E-6    # (A, D): уровни 1 и 4 -> лаг 3
    @test r[2, 3] / σ² ≈ θ[2] atol = 1E-6    # (C, D): уровни 3 и 4 -> лаг 1
    @test isposdef(r)
end

########################################################################
@testset "   9. веса наблюдений                                     " begin
########################################################################

    for (ct, ref) in ((Metida.TOEP, 15.2611541581), (Metida.TOEPH, 13.0580174493))
        l = memo((:wts, ct), () -> Metida.LMM(@formula(var ~ sequence + formulation), df0;
            repeated = Metida.VarEffect(Metida.@covstr(period | subject), ct), wts = "wts"))
        @test Metida.isfitted(l)
        @test Metida.m2logreml(l) ≈ ref atol = 1E-6
    end
end

########################################################################
@testset "  10. BAND(k) на базах помимо CSH                         " begin
########################################################################

    # Блок модификаторов проверяет BAND только на CSH. Здесь - остальные
    # базы, и главное: при k = 1 BAND СХЛОПЫВАЕТ различие между AR и CS
    # (и между ARH и CSH), потому что от структуры остаётся только лаг 1.
    # Обе пары обязаны сойтись на соответствующем TOEPP/TOEPHP(2).
    bmk(base, k) = memo((:band, base, k), () -> Metida.LMM(@formula(response ~ 1), ftdf3;
        repeated = [Metida.VarEffect(Metida.@covstr(r1 | subject), base),
                    Metida.VarEffect(Metida.@covstr(1 | subject), Metida.BAND(k))]))
    plain(base) = memo((:plain, base), () -> Metida.LMM(@formula(response ~ 1), ftdf3;
        repeated = Metida.VarEffect(Metida.@covstr(r1 | subject), base)))

    # гомогенные: AR + BAND(1) == CS + BAND(1) == TOEPP(2)
    @test Metida.m2logreml(bmk(Metida.AR, 1)) ≈ 974.75809 atol = 1E-5
    @test Metida.m2logreml(bmk(Metida.AR, 1)) ≈ Metida.m2logreml(bmk(Metida.CS, 1)) atol = 1E-6
    @test Metida.m2logreml(bmk(Metida.AR, 1)) ≈ Metida.m2logreml(plain(Metida.TOEPP(2))) atol = 1E-6
    # гетерогенные: ARH + BAND(1) == CSH + BAND(1) == TOEPHP(2)
    @test Metida.m2logreml(bmk(Metida.ARH, 1)) ≈ 963.05846 atol = 1E-5
    @test Metida.m2logreml(bmk(Metida.ARH, 1)) ≈ Metida.m2logreml(bmk(Metida.CSH, 1)) atol = 1E-6
    @test Metida.m2logreml(bmk(Metida.ARH, 1)) ≈ Metida.m2logreml(plain(Metida.TOEPHP(2))) atol = 1E-6

    # k = 2 различает базы снова
    @test Metida.m2logreml(bmk(Metida.ARH, 2)) ≈ 912.21798 atol = 1E-5
    @test Metida.m2logreml(bmk(Metida.CSH, 2)) ≈ 934.84359 atol = 1E-5
    @test !isapprox(Metida.m2logreml(bmk(Metida.ARH, 2)),
                    Metida.m2logreml(bmk(Metida.CSH, 2)), atol = 1E-4)

    # k не меньше размера блока - маска ничего не отсекает, и это ровно
    # базовая структура, с тем же числом параметров
    for base in (Metida.AR, Metida.ARH, Metida.CS, Metida.CSH)
        lb = bmk(base, 6); lp = plain(base)
        @test Metida.m2logreml(lb) ≈ Metida.m2logreml(lp) atol = 1E-6
        @test Metida.thetalength(lb) == Metida.thetalength(lp)
        @test Metida.dof(lb) == Metida.dof(lp)
    end
end

########################################################################
@testset "  11. предусловия и ошибки                                " begin
########################################################################

    # Уровневые структуры строят блок как rz * C * rz', поэтому две строки
    # одного уровня получают ковариацию, равную дисперсии этого уровня, и R
    # вырождается. Предупреждение обязано быть при конструировании, а не
    # падение посреди подгонки.
    for ct in (Metida.TOEP, Metida.TOEPH, Metida.UN)
        @test_logs (:warn,) match_mode = :any Metida.LMM(
            @formula(var ~ sequence + period + formulation), df0;
            repeated = Metida.VarEffect(Metida.@covstr(formulation | subject), ct))
    end
    # при уникальных уровнях - тихо
    for ct in (Metida.TOEP, Metida.TOEPH, Metida.UN)
        @test_nowarn Metida.LMM(@formula(var ~ sequence + formulation), df0;
            repeated = Metida.VarEffect(Metida.@covstr(period | subject), ct))
    end
    # параметризованные варианты предусловия не имеют и молчат всегда
    for ct in (Metida.TOEPP(2), Metida.TOEPHP(2))
        @test_nowarn Metida.LMM(@formula(var ~ sequence + period + formulation), df0;
            repeated = Metida.VarEffect(Metida.@covstr(formulation | subject), ct))
    end
    # использование в качестве СЛУЧАЙНОГО эффекта не затронуто
    for (ct, ref) in ((Metida.TOEP, 743.49838), (Metida.TOEPH, 737.37415))
        l = memo((:rnd, ct), () -> Metida.LMM(@formula(response ~ 1), ftdf3;
            random = Metida.VarEffect(Metida.@covstr(r1 | subject), ct)))
        @test Metida.isfitted(l)
        @test Metida.m2logreml(l) ≈ ref atol = 1E-4
    end
end

########################################################################
@testset "  12. отображение и остальной API                         " begin
########################################################################

    l = f4(Metida.TOEPH)
    @test_nowarn show(io, l)
    @test_nowarn show(io, Metida.TOEP)
    @test_nowarn show(io, Metida.TOEPH)
    @test_nowarn show(io, l.covstr)
    @test_nowarn Metida.rmatrix(l, 1)
    @test_nowarn Metida.raneff(l, 1)
    @test_nowarn Metida.rand(StableRNG(1), l)
    @test_nowarn Metida.coeftable(l)
    @test length(Metida.rand(StableRNG(1), l)) == Metida.nobs(l)
end

end
