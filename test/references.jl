################################################################################
# Словарь эталонов для тестов.
################################################################################

using CSV, DataFrames, Test

const REFDIR    = joinpath(@__DIR__, "references")
const REFVALUES = CSV.read(joinpath(REFDIR, "values.csv"), DataFrame;
                           types = Dict(:key => String, :value => String))
const REFMODELS = CSV.read(joinpath(REFDIR, "models.csv"), DataFrame)
const REFEFFECTS = CSV.read(joinpath(REFDIR, "effects.csv"), DataFrame)

# Значение хранится текстом ровно как в исходнике: целое остаётся целым,
# 1e-63 не теряет вид, строка приходит в кавычках, логическое - словом.
function _refparse(s)
    x = Meta.parse(strip(string(s)))
    x isa Union{Number,AbstractString,Bool} && return x
    x isa Expr && x.head === :call && x.args[1] === :- && length(x.args) == 2 &&
        x.args[2] isa Number && return -x.args[2]
    error("не удалось прочитать эталон: $s")
end

const _REFCACHE = Dict{String,Any}()

function _refload(key::AbstractString)
    haskey(_REFCACHE, key) && return _REFCACHE[key]
    rows = REFVALUES[REFVALUES.key .== key, :]
    isempty(rows) && error("нет эталона с ключом \"$key\" в test/references/values.csv")
    v = nrow(rows) == 1 && rows.index[1] == 0 ? _refparse(rows.value[1]) :
        [_refparse(rows.value[i]) for i in sortperm(rows.index)]
    _REFCACHE[key] = v
end

"""
    ref(key)

Эталонное значение по ключу из `test/references/values.csv`. Для векторного
эталона вернёт вектор; `refv` требует вектор явно.
"""
ref(key::AbstractString) = _refload(key)

"""
    refv(key)

То же, но всегда вектор: ключ с одним значением вернётся как вектор длины 1.
"""
refv(key::AbstractString) = (v = _refload(key); v isa AbstractVector ? v : [v])

function _reffield(key::AbstractString, col::Symbol)
    rows = REFVALUES[REFVALUES.key .== key, :]
    isempty(rows) && error("нет эталона с ключом \"$key\"")
    v = rows[1, col]
    (v === missing || v == "") ? nothing : v
end

"""
    reftol(key)

Допуск, с которым сверяется эталон, или `nothing` - тогда сравнение точное.
"""
reftol(key::AbstractString) = (v = _reffield(key, :atol); v === nothing ? nothing : _refparse(string(v)))

"""
    refrtol(key)

Относительный допуск или `nothing`.
"""
refrtol(key::AbstractString) = (v = _reffield(key, :rtol); v === nothing ? nothing : _refparse(string(v)))

"""
    refsource(key)

Откуда эталон: `metida` - собственное значение пакета, `spss`, `mixedmodels`,
`publication` - внешнее.
"""
refsource(key::AbstractString) = (v = _reffield(key, :source); v === nothing ? "" : string(v))

"""
    reforigin(key)

То же одним словом для журнала: `своё` или `внешнее`.
"""
reforigin(key::AbstractString) = refsource(key) == "metida" ? "своё" : "внешнее"

"""
    refdesc(key)

Описание проверки, если оно записано в словаре, иначе пустая строка.
"""
refdesc(key::AbstractString) =
    (v = hasproperty(REFVALUES, :description) ? _reffield(key, :description) : nothing;
     v === nothing ? "" : string(v))

"""
    refsection(key)

Путь тестсета, в котором стоит проверка.
"""
refsection(key::AbstractString) = (v = _reffield(key, :testset); v === nothing ? "" : string(v))

"""
    refquantity(key)

Величина (функция Metida), к которой относится эталон.
"""
refquantity(key::AbstractString) = (v = _reffield(key, :quantity); v === nothing ? "" : string(v))

"""
    refkeys(model = nothing)

Все ключи словаря или ключи одной модели - для сборки отчётных таблиц.
"""
refkeys(model = nothing) = model === nothing ? unique(REFVALUES.key) :
    unique(REFVALUES[coalesce.(REFVALUES.model_id, "") .== model, :key])

"""
    reftable(testset; model, quantity, source, wide = true)

Срез словаря для отчёта. `testset` - любая часть пути тестсета, как он
печатается в прогоне: `reftable("1. Конструирование")` вернёт эталоны этого
тестсета, `reftable("Publick API")` - вместе со всеми вложенными. Остальные
отборы - по модели, величине и источнику эталона.

`wide = true` (по умолчанию) даёт по строке на ключ, вектор - одной ячейкой;
`wide = false` - исходные строки, по одной на элемент вектора.

```julia
reftable("1. Конструирование модели")
reftable(; model = "m_toep_si_lmm")
reftable(; source = "spss", wide = false)
```
"""
function reftable(testset = nothing; model = nothing, quantity = nothing,
                  source = nothing, wide = true)
    t = REFVALUES
    # model_id пуст у ключей, которые не относятся ни к одной модели
    # (прямой вызов link-функции, например), поэтому missing - это "",
    # а не строка «missing».
    sel(col, v) = t = t[[occursin(v, x === missing ? "" : string(x)) for x in t[!, col]], :]
    testset  === nothing || sel(:testset, testset)
    model    === nothing || sel(:model_id, model)
    quantity === nothing || sel(:quantity, quantity)
    source   === nothing || sel(:source, source)
    wide || return t
    out = DataFrame(key = String[], model_id = String[], testset = String[],
                    quantity = String[], variant = String[], value = Any[],
                    atol = Any[], source = String[], line = Any[])
    for g in groupby(t, :key; sort = false)
        push!(out, (string(g.key[1]),
                    g.model_id[1] === missing ? "" : string(g.model_id[1]),
                    string(g.testset[1]),
                    string(g.quantity[1]),
                    g.variant[1] === missing ? "" : string(g.variant[1]),
                    nrow(g) == 1 ? _refparse(g.value[1]) :
                        [_refparse(g.value[i]) for i in sortperm(g.index)],
                    g.atol[1] === missing ? "" : g.atol[1],
                    string(g.source[1]), g.line[1]))
    end
    out
end

"""
    reftestsets()

Пути тестсетов, у которых есть эталоны, со счётом ключей - оглавление словаря.
"""
function reftestsets()
    g = combine(groupby(REFVALUES, :testset), :key => (x -> length(unique(x))) => :keys,
                :model_id => (x -> length(unique(x))) => :models)
    sort!(g, :testset)
end

"""
    refmodel(id)

Строка спецификации модели из `models.csv` вместе с её эффектами из
`effects.csv`.
"""
function refmodel(id::AbstractString)
    m = REFMODELS[REFMODELS.model_id .== id, :]
    isempty(m) && error("нет модели \"$id\" в test/references/models.csv")
    (model = first(eachrow(m)), effects = REFEFFECTS[REFEFFECTS.model_id .== id, :])
end

################################################################################
#                            ЖУРНАЛ ПРОВЕРОК
#
# Проверка пишется так:
#
#   @refcheck "api_refmodel.nobs" Metida.nobs(lmm)
#
# Макрос берёт из словаря ожидаемое значение и допуск, сравнивает с полученным,
# разворачивается в обычный `@test` (прогон видит его как любой другой) и
# добавляет запись в журнал. По окончании прогона журнал содержит по строке на
# каждую проверку: секцию, функцию, ожидаемое и полученное значение, источник
# эталона и описание.
################################################################################

const REFLOG = NamedTuple[]

# сравнение: допуск из словаря; без допуска - точное равенство
function _refequal(expected, actual, atol, rtol)
    if atol === nothing && rtol === nothing
        return expected == actual
    end
    kw = Dict{Symbol,Any}()
    atol === nothing || (kw[:atol] = atol)
    rtol === nothing || (kw[:rtol] = rtol)
    try
        return isapprox(actual, expected; kw...)
    catch
        return expected == actual
    end
end

_refdiff(expected, actual) =
    try
        if expected isa Number && actual isa Number
            Float64(actual - expected)
        elseif expected isa AbstractVector && actual isa AbstractVector &&
               length(expected) == length(actual) && eltype(expected) <: Number
            maximum(abs.(float.(actual) .- float.(expected)))
        else
            missing
        end
    catch
        missing
    end

_refshow(v) = v isa AbstractVector ? "[" * join(string.(v), ", ") * "]" : string(v)

# Секция берётся из живого стека тестсетов: это то место, где проверка
# действительно стоит, а не то, где записан ключ.
function _livesection()
    try
        ts = get(task_local_storage(), :__BASETESTNEXT__, nothing)
        ts === nothing && return ""
        return join([strip(string(t.description)) for t in ts
                     if hasproperty(t, :description)], " / ")
    catch
        return ""
    end
end

function _refcheck(key::AbstractString, actual, expr::AbstractString = "",
                   file::AbstractString = "", line::Integer = 0,
                   note::AbstractString = "")
    expected = ref(key)
    atol, rtol = reftol(key), refrtol(key)
    ok = _refequal(expected, actual, atol, rtol)
    push!(REFLOG, (n = length(REFLOG) + 1,
                   section     = (s = _livesection(); isempty(s) ? refsection(key) : s),
                   func        = refquantity(key),
                   key         = key,
                   expected    = _refshow(expected),
                   actual      = _refshow(actual),
                   diff        = _refdiff(expected, actual),
                   atol        = atol === nothing ? missing : atol,
                   source      = refsource(key),
                   origin      = reforigin(key),
                   description = refdesc(key),
                   note        = note,
                   status      = ok ? "pass" : "FAIL",
                   expr        = expr,
                   file        = basename(file),
                   line        = line))
    ok || @warn "эталон не совпал" key expected = _refshow(expected) actual = _refshow(actual) atol
    ok
end

"""
    @refcheck "ключ" выражение [пояснение]

Сверить выражение с эталоном из словаря и записать проверку в журнал.
Разворачивается в `@test`, поэтому в прогоне выглядит как обычная проверка.
Необязательное пояснение попадает в журнал отдельной колонкой: оно про эту
проверку, тогда как описание в словаре - про сам эталон.
"""
macro refcheck(key, expr, note = "")
    src  = string(expr)
    file = string(__source__.file)
    line = __source__.line
    quote
        Test.@test _refcheck($(esc(key)), $(esc(expr)), $src, $file, $line, $(esc(note)))
    end
end

# Имя функции для колонки `func`: сначала ищется вызов, явно помеченный
# `Metida.`, иначе берётся первый вызов выражения. Без этого
# `length(Metida.coef(lrd))` попадало бы в журнал как `length`, хотя проверка
# про `coef`. Если вызова нет совсем (сравнение вида `tt.df[2] > 2.0`), колонка
# остаётся пустой: само выражение уже лежит в `expr`, и подставлять его сюда
# значит ломать таблицу строкой на полсотни знаков.
function _reffunc(src)
    for re in (r"(?<![A-Za-z0-9_.])Metida\.(\w+)[(!]", r"(?<![A-Za-z0-9_.])(\w+)[(!]")
        m = match(re, src)
        m === nothing || return String(m.captures[1])
    end
    ""
end

# соотношение между двумя вычисленными величинами: эталона в словаре нет,
# ожидаемое значение считается в том же прогоне
function _refrel(desc::AbstractString, actual, expected, atol,
                 lhs::AbstractString, rhs::AbstractString,
                 func::AbstractString, file::AbstractString, line::Integer)
    ok = _refequal(expected, actual, atol, nothing)
    push!(REFLOG, (n = length(REFLOG) + 1,
                   section     = (s = _livesection(); isempty(s) ? "" : s),
                   func        = func,
                   key         = "",
                   expected    = _refshow(expected),
                   actual      = _refshow(actual),
                   diff        = _refdiff(expected, actual),
                   atol        = atol === nothing ? missing : atol,
                   source      = "relation",
                   origin      = "вычислено",
                   description = desc,
                   note        = rhs,
                   status      = ok ? "pass" : "FAIL",
                   expr        = lhs,
                   file        = basename(file),
                   line        = line))
    ok || @warn "соотношение не выполнено" desc expected = _refshow(expected) actual = _refshow(actual)
    ok
end

"""
    @refrel "описание" левое правое [допуск]

Проверить соотношение между двумя вычисленными величинами - то, у чего нет
эталона в словаре: `logreml = -0.5 · m2logreml`, `lmm.θ = theta(lmm)`. В журнал
идёт как запись с источником `вычислено`.
"""
macro refrel(desc, lhs, rhs, atol = nothing)
    ls, rs = string(lhs), string(rhs)
    fn = _reffunc(ls)                                  # нет вызова - само выражение
    file = string(__source__.file)
    line = __source__.line
    quote
        Test.@test _refrel($(esc(desc)), $(esc(lhs)), $(esc(rhs)), $(esc(atol)),
                           $ls, $rs, $fn, $file, $line)
    end
end

# свойство, которое должно выполняться: эталона нет, сравнивать не с чем
function _refprop(desc::AbstractString, ok, expr::AbstractString,
                  func::AbstractString, file::AbstractString, line::Integer)
    held = ok === true || ok == true
    push!(REFLOG, (n = length(REFLOG) + 1,
                   section     = (s = _livesection(); isempty(s) ? "" : s),
                   func        = func,
                   key         = "",
                   expected    = "true",
                   actual      = string(ok),
                   diff        = missing,
                   atol        = missing,
                   source      = "property",
                   origin      = "свойство",
                   description = desc,
                   note        = "",
                   status      = held ? "pass" : "FAIL",
                   expr        = expr,
                   file        = basename(file),
                   line        = line))
    held || @warn "свойство не выполнено" desc expr
    held
end

"""
    @refprop "описание" выражение

Проверить свойство, у которого нет ни эталона, ни парного значения:
`istriu(θ₂)`, `isposdef(Symmetric(θ₂))`, признак «без ошибок». В журнал идёт
записью с источником `свойство`.
"""
macro refprop(desc, expr)
    src  = string(expr)
    fn   = _reffunc(src)
    file = string(__source__.file)
    line = __source__.line
    quote
        Test.@test _refprop($(esc(desc)), $(esc(expr)), $src, $fn, $file, $line)
    end
end

# поведение вызова: ни эталона, ни парного значения, ни булева выражения -
# проверяется сам факт того, что вызов прошёл так, как должен
function _refbehave(desc::AbstractString, ok::Bool, expected::AbstractString,
                    actual::AbstractString, kind::AbstractString,
                    expr::AbstractString, func::AbstractString,
                    file::AbstractString, line::Integer)
    push!(REFLOG, (n = length(REFLOG) + 1,
                   section     = (s = _livesection(); isempty(s) ? "" : s),
                   func        = func,
                   key         = "",
                   expected    = expected,
                   actual      = actual,
                   diff        = missing,
                   atol        = missing,
                   source      = kind,
                   origin      = "поведение",
                   description = desc,
                   note        = "",
                   status      = ok ? "pass" : "FAIL",
                   expr        = expr,
                   file        = basename(file),
                   line        = line))
    ok || @warn "поведение не совпало" desc expected actual
    ok
end

_reftrunc(s, n = 200) = (t = strip(replace(string(s), r"\s+" => " ")); length(t) <= n ? t : t[1:n] * "…")

# выполнить f(), вернув (значение, текст stderr); исключение - в текст
function _refcapture(f)
    fname = tempname()
    try
        v = open(fname, "w") do io
            redirect_stderr(io) do
                f()
            end
        end
        return (v, read(fname, String), nothing)
    catch e
        err = read(fname, String)
        return (nothing, err, e)
    finally
        rm(fname, force = true)
    end
end

"""
    @refnowarn "описание" выражение

Вызов должен пройти без ошибок и без предупреждений. Эталона нет: проверяется
поведение, и в журнал идёт запись с источником `поведение`. Возвращает значение
выражения, как и `@test_nowarn`.
"""
macro refnowarn(desc, expr)
    src  = string(expr)
    fn   = _reffunc(src)
    file = string(__source__.file)
    line = __source__.line
    quote
        local _v, _err, _ex = _refcapture(() -> $(esc(expr)))
        local _ok = _ex === nothing && isempty(_err)
        Test.@test _refbehave($(esc(desc)), _ok, "без ошибок и предупреждений",
                              _ex !== nothing ? _reftrunc(sprint(showerror, _ex)) :
                              isempty(_err) ? "без ошибок и предупреждений" : _reftrunc(_err),
                              "nowarn", $src, $fn, $file, $line)
        _v
    end
end

"""
    @refwarn "описание" "ожидаемое предупреждение" выражение

Вызов должен выдать предупреждение, содержащее заданный текст. Возвращает
значение выражения, как и `@test_warn`, - поэтому годится там, где результат
вызова нужен дальше.
"""
macro refwarn(desc, msg, expr)
    src  = string(expr)
    fn   = _reffunc(src)
    file = string(__source__.file)
    line = __source__.line
    quote
        local _m = $(esc(msg))
        local _v, _err, _ex = _refcapture(() -> $(esc(expr)))
        local _ok = _ex === nothing && occursin(_m isa Regex ? _m : string(_m), _err)
        Test.@test _refbehave($(esc(desc)), _ok, "предупреждение: " * _reftrunc(_m, 120),
                              _ex !== nothing ? _reftrunc(sprint(showerror, _ex)) :
                              isempty(_err) ? "без предупреждений" : _reftrunc(_err),
                              "warn", $src, $fn, $file, $line)
        _v
    end
end

"""
    @refwarnthrows "описание" сообщение ТипИсключения выражение

Вызов должен ВЫДАТЬ предупреждение И бросить исключение заданного типа - обе
вещи на одном вызове.
"""
macro refwarnthrows(desc, msg, extype, expr)
    src  = string(expr)
    fn   = _reffunc(src)
    file = string(__source__.file)
    line = __source__.line
    quote
        local _m = $(esc(msg))
        local _T = $(esc(extype))
        local _v, _err, _ex = _refcapture(() -> $(esc(expr)))
        local _warned = occursin(_m isa Regex ? _m : string(_m), _err)
        local _threw  = _ex !== nothing && _ex isa _T
        local _got = string(_warned ? "предупреждение есть" : "предупреждения нет", ", ",
                            _ex === nothing ? "исключения не было" : string(typeof(_ex)))
        Test.@test _refbehave($(esc(desc)), _warned && _threw,
                              "предупреждение «" * _reftrunc(_m, 80) * "» и " * string(_T),
                              _got, "warn", $src, $fn, $file, $line)
    end
end

"""
    @refthrows "описание" ТипИсключения выражение

Вызов должен бросить исключение заданного типа. Для раздела «Обработка ошибок»
это единственная форма проверки: ожидаемого значения у неё нет, есть ожидаемое
поведение.
"""
macro refthrows(desc, extype, expr)
    src  = string(expr)
    fn   = _reffunc(src)
    file = string(__source__.file)
    line = __source__.line
    quote
        local _T = $(esc(extype))
        local _ok = false
        local _got = "исключения не было"
        try
            $(esc(expr))
        catch e
            _got = string(typeof(e))
            _ok  = e isa _T
        end
        Test.@test _refbehave($(esc(desc)), _ok, string(_T), _got,
                              "throws", $src, $fn, $file, $line)
    end
end

"""
    reflog()

Журнал проверок таблицей: по строке на каждую `@refcheck`.
"""
reflog() = isempty(REFLOG) ? DataFrame() : DataFrame(REFLOG)

"""
    reflogwrite(path = "test/references/testlog.csv")

Записать журнал в CSV и вернуть путь.
"""
function reflogwrite(path = joinpath(REFDIR, "testlog.csv"))
    isempty(REFLOG) && return path
    CSV.write(path, reflog())
    path
end

"""
    reflogsummary()

Короткая сводка по журналу: сколько записей, сколько прошло, по сколько на
секцию и по источнику эталона.
"""
function reflogsummary()
    t = reflog()
    isempty(t) && return "журнал пуст"
    io = IOBuffer()
    println(io, "Журнал проверок: ", nrow(t), " записей, прошло ",
            count(==("pass"), t.status), ", не прошло ", count(==("FAIL"), t.status), ".")
    println(io, "Источник: ",
            join([string(k, " - ", v) for (k, v) in
                  zip(keys(countmap_(t.origin)), values(countmap_(t.origin)))], ", "), ".")
    for g in groupby(t, :section)
        println(io, "  ", rpad(g.section[1], 62), lpad(nrow(g), 3), " проверок")
    end
    String(take!(io))
end

countmap_(xs) = (d = Dict{Any,Int}(); for x in xs; d[x] = get(d, x, 0) + 1; end; d)
