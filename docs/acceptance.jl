################################################################################
#                      ПРИЕМКА ВАЛИДАЦИОННОГО ОТЧЕТА
################################################################################

using CSV, DataFrames, Test, Printf

const ACC_TESTDIR = @__DIR__
const ACC_REFDIR  = joinpath(ACC_TESTDIR, "references")
const ACC_DOCSSRC = joinpath(ACC_TESTDIR, "..", "docs", "src")

# Критерий одного сравнения:
#
#     |x − x_ref| ≤ rtol · |x_ref| + atol
#
const ACCEPT_RTOL = 0.0
const ACCEPT_ATOL = 1.0E-6

const ACC_REQUIRED = ["API-JOURNAL", "PKG-TEST", "VAL-INIT", "VAL-MM", "VAL-BE",
                      "VAL-SPSS", "CONCLUSION"]

################################################################################
#                                  запись
################################################################################

struct AccRow
    scenario::String
    kind::String            # "comparison" | "outcome"
    quantity::String
    variant::String
    value::Float64
    reference::Float64
    difference::Float64
    relerr::Float64
    limit::Float64
    atol::Float64
    rtol::Float64
    evidence::String        # "external" | "analytic" | "internal" | "outcome"
    what::String
    status::String
end

const ACCEPTANCE = AccRow[]

"""
    acceptance_record(scenario, quantity, variant, value, reference; atol, rtol, evidence)

Записать одно сравнение и вернуть `"PASS"` или `"FAIL"`.
"""
function acceptance_record(scenario, quantity, variant, value, reference;
                           atol = ACCEPT_ATOL, rtol = ACCEPT_RTOL,
                           evidence = "external")
    v, r = float(value), float(reference)
    d    = abs(v - r)
    lim  = rtol * abs(r) + atol
    ok   = isfinite(d) && d <= lim
    push!(ACCEPTANCE, AccRow(string(scenario), "comparison", string(quantity),
          string(variant), v, r, d, iszero(r) ? NaN : d / abs(r), lim, atol, rtol,
          string(evidence), "", ok ? "PASS" : "FAIL"))
    ok ? "PASS" : "FAIL"
end

"""
    acceptance_mark(scenario, ok, what)

Записать исход, который не является сравнением с числом
"""
function acceptance_mark(scenario, ok::Bool, what; evidence = "outcome")
    push!(ACCEPTANCE, AccRow(string(scenario), "outcome", "", "", NaN, NaN, NaN,
          NaN, NaN, NaN, NaN, string(evidence), string(what), ok ? "PASS" : "FAIL"))
    ok
end

acc_rows(scenario) = filter(r -> r.scenario == string(scenario), ACCEPTANCE)

"""
    acceptance_note(scenario)

Строка под таблицу раздела
"""
function acceptance_note(scenario)
    rs = acc_rows(scenario)
    isempty(rs) && return "**Scenario $(scenario) did not report.**"
    cmp  = count(r -> r.kind == "comparison", rs)
    bad  = count(r -> r.status == "FAIL", rs)
    out  = filter(r -> r.kind == "outcome", rs)
    s = "Comparisons: $(cmp). Recorded outcomes: $(length(out)). Failed: $(bad)."
    for r in out
        s *= "\n\n  * " * (r.status == "PASS" ? "" : "**FAILED** - ") * r.what
    end
    s
end

################################################################################
#                      тестсет
################################################################################

mutable struct AccTestSet <: Test.AbstractTestSet
    description::String
    scenario::String
    toplevel::Bool
    pass::Int
    fail::Int
    err::Int
    broken::Int
    failures::Vector{String}
end

# `@testset` 
function AccTestSet(desc::AbstractString; scenario = nothing)
    if isnothing(scenario)
        p = Test.get_testset_depth() > 0 ? Test.get_testset() : nothing
        isa(p, AccTestSet) ?
            AccTestSet(string(desc), p.scenario, false, 0, 0, 0, 0, String[]) :
            AccTestSet(string(desc), string(desc), true, 0, 0, 0, 0, String[])
    else
        AccTestSet(string(desc), string(scenario), true, 0, 0, 0, 0, String[])
    end
end

_accbrief(t) = first(split(replace(sprint(show, t), r"\s+" => " "), " Stacktrace"))

Test.record(ts::AccTestSet, t::Test.Pass)   = (ts.pass   += 1; t)
Test.record(ts::AccTestSet, t::Test.Broken) = (ts.broken += 1; t)
Test.record(ts::AccTestSet, t::Test.Fail)   =
    (ts.fail += 1; length(ts.failures) < 50 && push!(ts.failures, _accbrief(t)); t)
Test.record(ts::AccTestSet, t::Test.Error)  =
    (ts.err  += 1; length(ts.failures) < 50 && push!(ts.failures, _accbrief(t)); t)

# 
function Test.record(ts::AccTestSet, child::AccTestSet)
    ts.pass += child.pass; ts.fail += child.fail
    ts.err  += child.err;  ts.broken += child.broken
    for f in child.failures; length(ts.failures) < 50 && push!(ts.failures, f); end
    child
end

function Test.record(ts::AccTestSet, child::Test.AbstractTestSet)
    _accwalk!(ts, child)
    child
end

function _accwalk!(ts::AccTestSet, child::Test.AbstractTestSet)
    isa(child, AccTestSet) && return Test.record(ts, child)
    hasproperty(child, :results) || return ts
    for r in child.results
        if isa(r, Test.Pass);        ts.pass   += 1
        elseif isa(r, Test.Broken);  ts.broken += 1
        elseif isa(r, Test.Fail);    ts.fail   += 1
            length(ts.failures) < 50 && push!(ts.failures, _accbrief(r))
        elseif isa(r, Test.Error);   ts.err    += 1
            length(ts.failures) < 50 && push!(ts.failures, _accbrief(r))
        elseif isa(r, Test.AbstractTestSet); _accwalk!(ts, r)
        end
    end
    # 
    isa(child, Test.DefaultTestSet) && hasproperty(child, :n_passed) &&
        (ts.pass += child.n_passed)
    ts
end

function Test.finish(ts::AccTestSet)
    if ts.toplevel
        acceptance_mark(ts.scenario, ts.fail == 0 && ts.err == 0,
            @sprintf("%s: %d passed, %d failed, %d errored, %d broken",
                     ts.description, ts.pass, ts.fail, ts.err, ts.broken))
        for f in ts.failures
            acceptance_mark(ts.scenario, false, "failed: " * f)
        end
    end
    Test.get_testset_depth() > 0 && Test.record(Test.get_testset(), ts)
    ts
end

"""
    acceptance_run(scenario, description) do ... end

Выполнить блок под счетным тестсетом.
"""
function acceptance_run(f, scenario, description)
    try
        Test.@testset AccTestSet scenario=string(scenario) "$description" begin
            f()
        end
    catch err
        acceptance_mark(scenario, false, "$description: run aborted - " *
            first(split(sprint(showerror, err), "\n")))
    end
    nothing
end

################################################################################
#                       журнал проверок API и словарь
################################################################################

const _ACC_CACHE = Dict{String,DataFrame}()

function _acc_csv(path)
    haskey(_ACC_CACHE, path) && return _ACC_CACHE[path]
    _ACC_CACHE[path] = CSV.read(path, DataFrame)
end

"""
    journal()

Журнал проверок `test.jl`: по строке на проверку.
"""
journal() = _acc_csv(joinpath(ACC_REFDIR, "testlog.csv"))

"""
    dictionary()

Словарь эталонов: по строке на (ключ, индекс).
"""
dictionary() = _acc_csv(joinpath(ACC_REFDIR, "values.csv"))

models_csv()  = _acc_csv(joinpath(ACC_REFDIR, "models.csv"))
effects_csv() = _acc_csv(joinpath(ACC_REFDIR, "effects.csv"))

_s(x) = ismissing(x) ? "" : string(x)

# 
const ACC_ORIGIN = [
 ("свое",      "metida",   "A dictionary reference that is the package's own value, carried forward. Establishes that the behaviour has not changed and **nothing more**."),
 ("внешнее",   "spss",     "A dictionary reference taken from IBM SPSS Statistics 28: agreement with an independent implementation."),
 ("вычислено", "relation", "A relation between two quantities computed in the same run: `aic = -2logml + 2p`, the ordering of two intervals, two routes to one number. No stored constant, so nothing can go stale."),
 ("свойство",  "property", "A claim with nothing to compare against: `isposdef`, symmetry, rank, a row count, a flag for non-estimability."),
 ("поведение", "nowarn",   "The call completes with no error and no warning."),
 ("поведение", "warn",     "The call emits a warning containing a given text."),
 ("поведение", "throws",   "The call throws an exception of a given type."),
]

"""
    journal_origin_table()

Журнал по роду утверждения
"""
function journal_origin_table()
    j = journal()
    df = DataFrame("Kind" => String[], "Source" => String[], "Records" => Int[],
                   "What it establishes" => String[])
    for (origin, source, meaning) in ACC_ORIGIN
        n = count(i -> _s(j.origin[i]) == origin && _s(j.source[i]) == source, 1:nrow(j))
        n == 0 && continue
        push!(df, (origin, source, n, meaning))
    end
    push!(df, ("total", "", nrow(j), ""))
    df
end

# 
const ACC_SECTION_EN = Dict(
 "Publick API basic tests / 1. Конструирование модели"                  => "1. Model construction",
 "Publick API basic tests / 2. Подгонка: значения REML и ML"             => "2. Fitting: REML and ML values",
 "Publick API basic tests / 3. Внутренние величины REML"                 => "3. Internal REML quantities",
 "Publick API basic tests / 4. Интерфейс StatsAPI / StatsBase"           => "4. StatsAPI / StatsBase interface",
 "Publick API basic tests / 5. Матрицы модели"                           => "5. Model matrices",
 "Publick API basic tests / 6. Инференс: DF, контрасты, тип III"         => "6. Inference: DF, contrasts, Type III",
 "Publick API basic tests / 7. Link-функции и стратегии первого шага"    => "7. Link functions and first-step strategies",
 "Publick API basic tests / 8. Потоки и воспроизводимость"               => "8. Threads and reproducibility",
 "Publick API basic tests / 9. Веса наблюдений"                          => "9. Observation weights",
 "Publick API basic tests / 10. Краевые случаи"                          => "10. Edge cases",
 "Publick API basic tests / 11. Логирование и отображение"               => "11. Logging and display",
 "Publick API basic tests / 12. Обработка ошибок"                        => "12. Error handling",
 "Publick API basic tests / 13. Симуляция и ресемплинг"                  => "13. Simulation and resampling",
 "Publick API basic tests / 14. Предельные средние: lsmeans и emmeans"   => "14. Marginal means: lsmeans and emmeans",
)

"""
    journal_section_table(; prefix = nothing)

Журнал по секциям в порядке прохождения. `prefix` оставляет только секции,
начинающиеся с него.
"""
function journal_section_table(; prefix = nothing)
    j = journal()
    secs = String[]
    for s in _s.(j.section)
        isempty(s) && continue
        (isnothing(prefix) || startswith(s, prefix)) || continue
        s in secs || push!(secs, s)
    end
    df = DataFrame("Section" => String[], "Records" => Int[], "Passed" => Int[],
                   "Failed" => Int[])
    for s in secs
        idx = findall(i -> _s(j.section[i]) == s, 1:nrow(j))
        push!(df, (get(ACC_SECTION_EN, s, replace(s, r"^Publick API basic tests / " => "")), length(idx),
                   count(i -> _s(j.status[i]) == "pass", idx),
                   count(i -> _s(j.status[i]) != "pass", idx)))
    end
    df
end

"""
    dictionary_source_table()

Словарь по источнику эталона.
"""
function dictionary_source_table()
    d = dictionary()
    df = DataFrame("Source" => String[], "Keys" => Int[], "Rows" => Int[],
                   "What it is" => String[])
    meaning = Dict(
      "metida"       => "A value of the package itself, fixed as a regression reference",
      "spss"         => "IBM SPSS Statistics 28",
      "mixedmodels"  => "MixedModels.jl, in the same run",
      "publication"  => "A published table")
    for g in groupby(d, :source)
        src = _s(g.source[1])
        push!(df, (src, length(unique(g.key)), nrow(g), get(meaning, src, "")))
    end
    sort!(df, "Rows", rev = true)
    push!(df, ("total", length(unique(d.key)), nrow(d), ""))
    df
end

"""
    api_name_table(mod)

Публичные имена модуля и число записей журнала, в которых проверка названа по
этому имени. 
"""
function api_name_table(mod)
    j     = journal()
    funcs = _s.(j.func)
    blob  = join(funcs, "\n") * "\n" * join(_s.(j.expr), "\n")
    elsewhere = Dict(
      "ACOV" => "test_acov_band_block.jl", "BAND" => "test_acov_band_block.jl",
      "SPEXP" => "test.jl, Spatial section", "SPGAU" => "test.jl, Spatial section",
      "SPPOW" => "test.jl, Spatial section", "RawCoding" => "test_new_structures.jl",
      "bootstrap" => "test.jl §13, recorded under `confint`",
      "coeftable" => "test.jl §4, recorded under `coef`; regressions.jl",
      "vmatrix!" => "test.jl §5, recorded under `vmatrix`",
      "ARH" => "models.csv: 4 models", "CSH" => "models.csv: 19 models")
    df = DataFrame("Name" => String[], "Records" => Int[], "Covered elsewhere" => String[])
    for n in sort(string.(names(mod)))
        n == string(nameof(mod)) && continue
        c = count(==(n), funcs)
        push!(df, (n, c, c > 0 ? "" :
                   get(elsewhere, n, occursin(n, blob) ? "in check expressions" : "-")))
    end
    df
end

"""
    structure_table()

Структуры ковариации по спецификациям моделей в `effects.csv`.
"""
function structure_table()
    e = effects_csv()
    df = combine(groupby(e, :structure), nrow => :n)
    sort!(df, :n, rev = true)
    rename!(df, :structure => "Structure", :n => "Effects")
    df[!, "Structure"] = _s.(df[!, "Structure"])
    push!(df, ("total", sum(df[!, "Effects"])))
    df
end

"""
    dataset_table()

Наборы данных по спецификациям моделей.
"""
function dataset_table()
    m = models_csv()
    df = combine(groupby(m, :dataset), nrow => :n)
    sort!(df, :n, rev = true)
    rename!(df, :dataset => "Dataset", :n => "Models")
    df[!, "Dataset"] = [isempty(x) ? "(not recorded)" : x for x in _s.(df[!, "Dataset"])]
    push!(df, ("total", sum(df[!, "Models"])))
    df
end

################################################################################
#                               сводка и вердикт
################################################################################

"""
    acceptance_summary_table()

Разделы отчета, число сравнений и число непрошедших исходов. 
"""
function acceptance_summary_table()
    df = DataFrame("Scenario" => String[], "Comparisons" => Any[], "Outcomes" => Any[],
                   "Failed" => Any[], "Status" => String[])
    seen = String[]
    for r in ACCEPTANCE; r.scenario in seen || push!(seen, r.scenario); end
    for s in vcat(seen, [x for x in ACC_REQUIRED if !(x in seen)])
        rs = acc_rows(s)
        if isempty(rs)
            push!(df, (s, "-", "-", "-", "NOT REPORTED")); continue
        end
        bad = count(r -> r.status == "FAIL", rs)
        push!(df, (s, count(r -> r.kind == "comparison", rs),
                   count(r -> r.kind == "outcome", rs), bad,
                   bad == 0 ? "PASS" : "FAIL"))
    end
    df
end

# Непрошедшие строки журнала проверок
function journal_failures()
    j = journal()
    idx = findall(i -> _s(j.status[i]) != "pass", 1:nrow(j))
    DataFrame("Section" => [_s(j.section[i]) for i in idx],
              "Check" => [_s(j.description[i]) for i in idx],
              "Key" => [_s(j.key[i]) for i in idx],
              "Expected" => [_s(j.expected[i]) for i in idx],
              "Actual" => [_s(j.actual[i]) for i in idx],
              "Tolerance" => [_s(j.atol[i]) for i in idx],
              "Source" => [_s(j.file[i]) * ":" * _s(j.line[i]) for i in idx])
end

"""
    acceptance_journal_block()

"""
function acceptance_journal_block()
    jf  = journal_failures()
    bad = filter(r -> r.status == "FAIL", ACCEPTANCE)
    miss = [s for s in ACC_REQUIRED if isempty(acc_rows(s))]
    s = ""
    s *= "**The API check journal.** " * (nrow(jf) == 0 ?
         "None of its $(nrow(journal())) records failed." :
         "$(nrow(jf)) of $(nrow(journal())) records failed:") * "\n\n"
    nrow(jf) > 0 && (s *= acceptance_md(jf) * "\n\n")
    s *= "**The report's own records.** " * (isempty(bad) ?
         "None of its $(length(ACCEPTANCE)) records failed." :
         "$(length(bad)) of $(length(ACCEPTANCE)) records failed:") * "\n\n"
    for r in bad
        s *= "  * `" * r.scenario * "` - " * (r.kind == "outcome" ? r.what :
             @sprintf("%s (%s): %.10g against %.10g, difference %.3g, limit %.3g",
                      r.quantity, r.variant, r.value, r.reference, r.difference, r.limit)) * "\n"
    end
    isempty(bad) || (s *= "\n")
    s *= "**Required scenarios.** " * (isempty(miss) ?
         "All $(length(ACC_REQUIRED)) reported." :
         "Did not report: " * join(miss, ", ") * ".")
    s
end

"""
    acceptance_md(df)

Напечатать DataFrame таблицей markdown.

"""
function acceptance_md(df::DataFrame)
    cols = names(df)
    s = "| " * join(cols, " | ") * " |\n|" * repeat("---|", length(cols)) * "\n"
    for r in eachrow(df)
        s *= "| " * join([replace(string(r[c]), "|" => "\\|") for c in cols], " | ") * " |\n"
    end
    s
end

"""
    acceptance_verdict()

`(ok, текст)`. Отрицательный вердикт дается, если не прошла строка журнала,
не прошла запись отчета или обязательный раздел не отчитался.
"""
function acceptance_verdict()
    jbad = nrow(journal_failures())
    jn   = nrow(journal())
    abad = count(r -> r.status == "FAIL", ACCEPTANCE)
    miss = [s for s in ACC_REQUIRED if isempty(acc_rows(s))]
    ok   = jbad == 0 && abad == 0 && isempty(miss) && jn > 0
    txt  = ok ?
      "**VERDICT: ACCEPTED.** API check journal: $(jn) records, 0 failed. " *
      "The report's own records: $(length(ACCEPTANCE)), 0 failed. " *
      "Required scenarios: $(length(ACC_REQUIRED)), all reported." :
      "**VERDICT: NOT ACCEPTED.** API check journal: $(jn) records, $(jbad) " *
      "failed. The report's own records: $(length(ACCEPTANCE)), $(abad) failed. " *
      "Scenarios that did not report: " *
      (isempty(miss) ? "none" : join(miss, ", ")) * ". Every item is listed below."
    (ok, txt)
end

acceptance_conclusion_text() = last(acceptance_verdict())

"""
    acceptance_write_journal()
"""
function acceptance_write_journal()
    isdir(ACC_DOCSSRC) || mkpath(ACC_DOCSSRC)
    full = DataFrame(scenario = [r.scenario for r in ACCEPTANCE],
                     kind = [r.kind for r in ACCEPTANCE],
                     quantity = [r.quantity for r in ACCEPTANCE],
                     variant = [r.variant for r in ACCEPTANCE],
                     value = [r.value for r in ACCEPTANCE],
                     reference = [r.reference for r in ACCEPTANCE],
                     difference = [r.difference for r in ACCEPTANCE],
                     relerr = [r.relerr for r in ACCEPTANCE],
                     limit = [r.limit for r in ACCEPTANCE],
                     evidence = [r.evidence for r in ACCEPTANCE],
                     what = [r.what for r in ACCEPTANCE],
                     status = [r.status for r in ACCEPTANCE])
    CSV.write(joinpath(ACC_DOCSSRC, "validation_acceptance.tsv"), full; delim = '\t')
    div = vcat(journal_failures(),
               DataFrame("Section" => [r.scenario for r in ACCEPTANCE if r.status == "FAIL"],
                         "Check" => [r.kind == "outcome" ? r.what : r.quantity
                                     for r in ACCEPTANCE if r.status == "FAIL"],
                         "Key" => ["" for r in ACCEPTANCE if r.status == "FAIL"],
                         "Expected" => [r.kind == "outcome" ? "" : string(r.reference)
                                        for r in ACCEPTANCE if r.status == "FAIL"],
                         "Actual" => [r.kind == "outcome" ? "" : string(r.value)
                                      for r in ACCEPTANCE if r.status == "FAIL"],
                         "Tolerance" => [r.kind == "outcome" ? "" : string(r.limit)
                                         for r in ACCEPTANCE if r.status == "FAIL"],
                         "Source" => ["the report" for r in ACCEPTANCE if r.status == "FAIL"]))
    CSV.write(joinpath(ACC_DOCSSRC, "validation_divergences.tsv"), div; delim = '\t')
    (joinpath(ACC_DOCSSRC, "validation_acceptance.tsv"),
     joinpath(ACC_DOCSSRC, "validation_divergences.tsv"))
end

"""
    acceptance_verdict_from_files(; acc, div, log)

"""
function acceptance_verdict_from_files(;
        acc = joinpath(ACC_DOCSSRC, "validation_acceptance.tsv"),
        log = joinpath(ACC_REFDIR,  "testlog.csv"))
    isfile(acc) || return (false, "**VERDICT: NOT ACCEPTED.** The report wrote no " *
        "acceptance log: $(acc) is missing, so no chapter reported.")
    isfile(log) || return (false, "**VERDICT: NOT ACCEPTED.** The API check journal " *
        "$(log) is missing: the test suite did not run.")
    a = CSV.read(acc, DataFrame; delim = '\t')
    j = CSV.read(log, DataFrame)
    abad = count(i -> _s(a.status[i]) == "FAIL", 1:nrow(a))
    jbad = count(i -> _s(j.status[i]) != "pass", 1:nrow(j))
    seen = Set(_s.(a.scenario))
    miss = [s for s in ACC_REQUIRED if !(s in seen)]
    ok   = abad == 0 && jbad == 0 && isempty(miss) && nrow(j) > 0
    txt  = ok ?
      "**VERDICT: ACCEPTED.** API check journal: $(nrow(j)) records, 0 failed. " *
      "Acceptance log: $(nrow(a)) records, 0 failed. Required scenarios: " *
      "$(length(ACC_REQUIRED)), all reported." :
      "**VERDICT: NOT ACCEPTED.** API check journal: $(nrow(j)) records, $(jbad) " *
      "failed. Acceptance log: $(nrow(a)) records, $(abad) failed. Scenarios that " *
      "did not report: " * (isempty(miss) ? "none" : join(miss, ", ")) * "."
    (ok, txt)
end
