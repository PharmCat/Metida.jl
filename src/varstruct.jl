const CType = Union{FunctionTerm{typeof(+), Vector{Term}}, FunctionTerm{typeof(*), Vector{Term}}, FunctionTerm{typeof(&), Vector{Term}}}

import StatsModels: ContrastsMatrix, AbstractContrasts, modelcols

"""
    mutable struct RawCoding <: AbstractContrasts

Contrast for CategoricalTerm to get column "as it is" for model matrix.
"""
mutable struct RawCoding <: AbstractContrasts
end
function StatsModels.ContrastsMatrix(contrasts::RawCoding, levels::AbstractVector{T}) where T
    ContrastsMatrix(ones(1,1),
                             ["levels"],
                             levels,
                             contrasts)
end
function StatsModels.modelcols(t::CategoricalTerm{RawCoding, T, N}, d::NamedTuple) where T where N
    return d[t.sym]
end

################################################################################
#                     @covstr macro
################################################################################

"""
    @covstr(ex)

Macros for random/repeated effect model.

# Example

```julia
@covstr(model|subject)
```
"""
macro covstr(ex)
    return :(@formula(nothing ~ $ex).rhs)
end
function modelparse(term::FunctionTerm{typeof(|)})
    eff, subj = term.args
    if !isa(subj, AbstractTerm) || isa(subj, FunctionTerm{typeof(*), Vector{Term}}) throw(FormulaException("Subject term type not <: AbstractTerm. Use `term` or `interaction term` only. Maybe you are using something like this: `@covstr(factor|term1*term2)` or `@covstr(factor|(term1+term2))`. Use only `@covstr(factor|term)` or `@covstr(factor|term1&term2)`.")) end
    return eff, subj
end
function modelparse(term)
    throw(FormulaException("Model term type not <: FunctionTerm{typeof(|)}. Use model like this: `@covstr(factor|subject)`. Maybe you are using something like this: `@covstr(factor|term1+term2)`. Use only `@covstr(factor|term)` or `@covstr(factor|term1&term2)`."))
end

################################################################################
#                  EFFECT
################################################################################
"""
    VarEffect(formula, covtype::T, coding) where T <: AbstractCovarianceType

    VarEffect(formula, covtype::T; coding = nothing) where T <: AbstractCovarianceType

    VarEffect(formula; coding = nothing)

Random/repeated effect.

* `formula` from @covstr(ex) macros.

* `covtype` - covariance type (SI, DIAG, CS, CSH, AR, ARH, ARMA, TOEP, TOEPH, TOEPP, TOEPHP)

!!! note

    Categorical factors are coded with `FullDummyCoding()` by default, use `coding` for other contrast coding.

# Example

```julia
VarEffect(@covstr(1+factor|subject), CSH)

VarEffect(@covstr(1 + formulation|subject), CSH; coding = Dict(:formulation => StatsModels.DummyCoding()))
```
"""
struct VarEffect
    formula::FunctionTerm
    model::Union{Tuple{Vararg{AbstractTerm}}, AbstractTerm}
    covtype::CovarianceType
    coding::Dict{Symbol, AbstractContrasts}
    subj::AbstractTerm
    p::Int
    function VarEffect(formula, covtype::CovarianceType, coding)
        model, subj = modelparse(formula)
        p = nterms(model)
        if coding === nothing
            coding = Dict{Symbol, AbstractContrasts}()
        end
        #if !isa(subj, Union{CategoricalTerm,ConstantTerm,InteractionTerm{<:NTuple{N,CategoricalTerm} where {N}},}) error("subject (blocking) variables must be Categorical") end
        new(formula, model, covtype, coding, subj, p)
    end
    function VarEffect(formula, covtype::CovarianceType; coding = nothing)
        VarEffect(formula, covtype, coding)
    end
    function VarEffect(formula, covtype::AbstractCovarianceType; coding = nothing)
        VarEffect(formula, CovarianceType(covtype), coding)
    end
    function VarEffect(formula; coding = nothing)
        VarEffect(formula, SI, coding)
    end
end
################################################################################
#                            COVARIANCE STRUCTURE
################################################################################
#=
function sabjcrossdicts(d1, d2)
    
    if length(d1) == 1 
        return d1 
    elseif length(d2) == 1 
        return d2 
    elseif length(d1) == 0 
        return d2 
    end
    d2 = copy(d2)
    d1 = copy(d1)
    i = 0
    v = Dict{Int, Vector{Int}}()
    while length(d2) > 0
        i   += 1
        fk   = first(keys(d2))
        v[i] = copy(d2[fk])
        delete!(d2, fk)
        for (k1, v1) in d1
            if any(x -> x in v[i], v1)
                if !(v1 ⊆ v[i])
                    sd = setdiff(v1, v[i])
                    if  length(sd) > 0
                        for (k2, v2) in d2
                            if any(x -> x in v2, sd)
                                append!(v[i], v2)
                                delete!(d2, k2)
                            end
                        end
                    end
                end
                delete!(d1, k1)
            end
        end
    end
    return v
end
=#
#= 
"""

    dsu_find!(p, i)
 
Union-Find (disjoint set union)
Корень компоненты, содержащей элемент `i`
path halving
"""
=#
@inline function dsu_find!(p::Vector{Int}, i::Int)
    @inbounds while p[i] != i
        p[i] = p[p[i]]        # path halving: подвешиваем к «деду»
        i    = p[i]
    end
    return i
end
#=
"""
    dsu_union!(p, sz, a, b)
 
Объединяет компоненты элементов `a` и `b`. Меньшая по размеру подвешивается
к большей (union by size)
"""
=#
@inline function dsu_union!(p::Vector{Int}, sz::Vector{Int}, a::Int, b::Int)
    ra = dsu_find!(p, a)
    rb = dsu_find!(p, b)
    ra === rb && return ra                    # уже в одной компоненте
    @inbounds begin
        if sz[ra] < sz[rb]
            ra, rb = rb, ra                   # подвешиваем меньшую к большей
        end
        p[rb]   = ra
        sz[ra] += sz[rb]
    end
    return ra
end
#=
"""
    _maxobsindex(ds...)
 
Максимальный номер наблюдения, встречающийся в переданных словарях.
"""
=#
function _maxobsindex(ds...)
    n = 0
    for d in ds
        for v in values(d)
            @inbounds for i in v
                i > n && (n = i)
            end
        end
    end
    return n
end
#=
"""
    _components(parent, sz, n)
 
Выделяет компоненты связности из готовой DSU-структуры.
 
Ключевые свойства результата:
  * ключи — плотный диапазон `1:nb`;
  * блоки нумеруются по возрастанию минимального номера наблюдения;
  * внутри блока наблюдения идут в порядке возрастания номера строки.
"""
=#
function _components(parent::Vector{Int}, sz::Vector{Int}, n::Int)
    blkid = zeros(Int, n)                     # корень -> номер блока
    vecs  = Vector{Vector{Int}}()             # сами блоки
    fptr  = Int[]                             # указатели заполнения блоков
    @inbounds for r in 1:n
        rt = dsu_find!(parent, r)
        b  = blkid[rt]
        if b == 0                             # новая компонента
            push!(vecs, Vector{Int}(undef, sz[rt]))   # точный размер
            push!(fptr, 1)
            b          = length(vecs)
            blkid[rt]  = b
            vecs[b][1] = r
        else
            fptr[b]         += 1
            vecs[b][fptr[b]] = r
        end
    end
    res = Dict{Int, Vector{Int}}()
    sizehint!(res, length(vecs))
    @inbounds for b in eachindex(vecs)
        res[b] = vecs[b]
    end
    return res
end
#=
"""
    _canonical_blocks(d)
 
Приводит произвольный словарь «субъект => номера наблюдений» к каноническому
виду `Dict{Int, Vector{Int}}` с ключами `1:nb`, упорядоченными по минимальному
номеру наблюдения, и отсортированными значениями.
"""
=#
function _canonical_blocks(d)
    ks   = collect(keys(d))
    mins = Vector{Int}(undef, length(ks))
    @inbounds for i in eachindex(ks)
        mins[i] = minimum(d[ks[i]])
    end
    res = Dict{Int, Vector{Int}}()
    sizehint!(res, length(ks))
    @inbounds for (b, j) in enumerate(sortperm(mins))
        v      = d[ks[j]]
        res[b] = issorted(v) ? collect(v) : sort(v)
    end
    return res
end
#=
"""
    subjcrossdicts(d1, d2 [, n])
 
Наименьшее общее огрубление двух разбиений множества наблюдений `1:n`:
каждая группа из `d1` и каждая группа из `d2` целиком лежит внутри одного
блока результата, и блоков при этом максимально много.
 
Эквивалентно поиску компонент связности графа, в котором наблюдения соединены
рёбрами внутри каждой группы `d1` и внутри каждой группы `d2`.
 
Возвращает `Dict{Int, Vector{Int}}` с ключами `1:nb`.
"""
=#
function subjcrossdicts(d1, d2, n::Int = _maxobsindex(d1, d2))
    # --- вырожденные случаи -------------------------------------------------
    isempty(d1) && return _canonical_blocks(d2)
    isempty(d2) && return _canonical_blocks(d1)
 
    if (length(d1) == 1 && length(first(values(d1))) == n) ||
       (length(d2) == 1 && length(first(values(d2))) == n)
        return Dict{Int, Vector{Int}}(1 => collect(1:n))
    end
    # --- union-find ---------------------------------------------------------
    parent = collect(1:n)
    sz     = ones(Int, n)
    for d in (d1, d2)
        for v in values(d)
            isempty(v) && continue
            r0 = first(v)                     # «якорь» группы
            @inbounds for r in v
                dsu_union!(parent, sz, r0, r) # первое объединение — no-op
            end
        end
    end
    return _components(parent, sz, n)
end
#=
"""
    crossdicts(dicts, inds, n)
Многосторонний вариант: сразу объединяет все группировки `dicts[i], i ∈ inds`.
"""
=#
function crossdicts(dicts, inds, n::Int)
    parent = collect(1:n)
    sz     = ones(Int, n)
    for i in inds
        for v in values(dicts[i])
            isempty(v) && continue
            r0 = first(v)
            @inbounds for r in v
                dsu_union!(parent, sz, r0, r)
            end
        end
    end
    return _components(parent, sz, n)
end



tabcols(data, symbs) = Tuple(Tables.getcolumn(Tables.columns(data), x) for x in symbs)

struct EffectSubjectBlock
    sblock::Matrix{Vector{Tuple{Vector{Int}, Int}}}
    snames::Vector{Any}
end
function getsubj(covstr, effn, block, sbjn)
    covstr.esb.sblock[block, effn][sbjn][1]
end
function getsubjnn(covstr, effn, block, sbjn)
    covstr.esb.sblock[block, effn][sbjn][2]
end
function getsubjname(covstr, i)
    covstr.esb.snames[i]
end
function subjn(covstr, effn, block)
    length(covstr.esb.sblock[block, effn])
end
"""
    Return number of subject foe each random effet in current block.
"""
function raneflenv(covstr, block)
    l = size(covstr.esb.sblock, 2) - 1
    v = Vector{Int}(undef, l)
    for i = 1:l
        v[i] = length(covstr.esb.sblock[block, i])
    end
    return v
end


"""
    make_effect_subject_block(dicts, blocks, alleffl, rown)
 
Строит `EffectSubjectBlock`: для каждой пары (блок ковариационной матрицы,
эффект) — список субъектов этого эффекта, попавших в блок, в виде
`(позиции внутри блока, сквозной номер субъекта)`.
 
Идея оптимизации: вместо того чтобы для каждой пары (блок, группа) искать
пересечение линейным поиском (`findall(x -> x in v, blocks[i])`, O(|блок|·|v|)),
один раз строятся обратные индексы `строка -> (блок, позиция в блоке)`, после
чего позиции любой группы получаются за O(|v|) прямым обращением.
 
Сложность: O(alleffl · rown + Σ G·log G) по времени (G — число субъектов
эффекта; логарифм даёт только детерминированная сортировка групп),
O(rown) дополнительной памяти. Было — O(alleffl · rown²).
 
Дополнительно исправлено:
  * порядок субъектов детерминирован (по минимальному номеру наблюдения),
    а не задаётся хеш-порядком `Dict` — раньше имена в `esb.snames` и нумерация
    в `raneff` могли меняться от запуска к запуску / версии Julia;
  * позиции внутри субъекта гарантированно возрастают (как давал `findall`) —
    от этого зависит порядок лагов в AR/TOEP и порядок точек в SP*-структурах;
  * `snames` типизирован явно (`Vector{Any}`) вместо `nblock = []`.
"""
function make_effect_subject_block(dicts, blocks::Vector{Vector{Int}},
                                   alleffl::Int, rown::Int)
    nb = length(blocks)
 
    # Обратные индексы: строка -> блок и строка -> позиция в блоке.
    # Нули означают - строка не попала ни в один блок, в норме не бывает
    blockof = zeros(Int, rown)
    posof   = zeros(Int, rown)
    @inbounds for b in 1:nb
        bl = blocks[b]
        for p in eachindex(bl)
            r          = bl[p]
            blockof[r] = b
            posof[r]   = p
        end
    end
 
    #  Преаллокация выходной матрицы: все ячейки должны быть заполнены 
    sblock = Matrix{Vector{Tuple{Vector{Int}, Int}}}(undef, nb, alleffl)
    @inbounds for s in 1:alleffl, b in 1:nb
        sblock[b, s] = Vector{Tuple{Vector{Int}, Int}}(undef, 0)
    end
 
    snames = Vector{Any}(undef, 0)            # имена (ключи) субъектов
    nli    = 0                                # сквозной номер субъекта
 
    # По одному линейному проходу на эффект.
    for s in 1:alleffl
        d  = dicts[s]
        ks = collect(keys(d))
        sizehint!(snames, length(snames) + length(ks))
 
        # детерминированный порядок групп
        mins = Vector{Int}(undef, length(ks))
        @inbounds for i in eachindex(ks)
            mins[i] = minimum(d[ks[i]])
        end
 
        for j in sortperm(mins)
            k = ks[j]
            v = d[k]
            isempty(v) && continue
 
            # Быстрый путь: вся группа лежит в одном блоке. Это верно для всех
            # эффектов, участвовавших в построении блокировки, т.е. практически всегда.
            b1     = blockof[first(v)]
            single = b1 != 0
            if single
                @inbounds for r in v
                    if blockof[r] != b1
                        single = false
                        break
                    end
                end
            end
 
            if single
                fa = Vector{Int}(undef, length(v))
                @inbounds for (i, r) in enumerate(v)
                    fa[i] = posof[r]
                end
                issorted(fa) || sort!(fa)     # обычно уже отсортировано
                nli += 1
                push!(sblock[b1, s], (fa, nli))
                push!(snames, k)
            else
                # Медленный путь: группа пересекает несколько блоков. Возникает
                # для эффектов, не участвовавших в блокировке,  прежде всего
                # для эффекта-заглушки RZero (`1|1`) в моделях без случайных
                # эффектов, где это ровно одна группа на весь набор данных.
                # Сортируем строки по (блок, позиция) и режем на серии.
                ord = sortperm(v; by = r -> (blockof[r], posof[r]))
                i, m = 1, length(ord)
                while i <= m
                    b  = blockof[v[ord[i]]]
                    j2 = i
                    @inbounds while j2 < m && blockof[v[ord[j2 + 1]]] == b
                        j2 += 1
                    end
                    if b != 0                 # b == 0 — строки вне блоков
                        fa = Vector{Int}(undef, j2 - i + 1)
                        @inbounds for t in i:j2
                            fa[t - i + 1] = posof[v[ord[t]]]
                        end
                        nli += 1
                        push!(sblock[b, s], (fa, nli))
                        push!(snames, k)
                    end
                    i = j2 + 1
                end
            end
        end
    end
 
    return EffectSubjectBlock(sblock, snames)
end

"""
    Covarince structure.
"""
struct CovStructure{T, T2} <: AbstractCovarianceStructure
    # Random effects
    random::Vector{VarEffect}
    # Repearted effects
    repeated::Vector{VarEffect}
    # schema
    schema::Vector{Union{Tuple, AbstractTerm}}
    # names
    rcnames::Vector{String}
    # blocks for vcov matrix / variance blocking factor (subject)
    vcovblock::Vector{Vector{Int}}
    # number of random effect 
    rn::Int
    # number coef. of random effect in θ vector
    rtn::Int
    # number of repeated effect
    rpn::Int
    # Z matrix
    z::Matrix{T}
    #subjz::Vector{BitArray{2}}
    # Blocks for each blocking subject, each effect, each effect subject sblock[block][rand eff][subj]
    #
    esb::EffectSubjectBlock
    # unit range z column range for each random effect
    zrndur::Vector{UnitRange{Int}}
    # repeated effect parametrization matrix
    rz::Vector{Matrix{T2}}
    # size 2 of z/rz matrix
    q::Vector{Int}
    # total number of parameters in each effect
    t::Vector{Int}
    # range of each parameters in θ vector
    tr::Vector{UnitRange{Int}}
    # θ Parameter count
    tl::Int
    # Parameter type :var / :rho
    ct::Vector{Symbol}
    # map i->j where i - number of paran in theta and j n umber of effect
    emap::Vector{Int}
    # Nubber of subjects in each effect
    sn::Vector{Int}
    # Maximum number per block
    maxn::Int
    #--
    function CovStructure(random, repeated, data)
        alleffl =  length(random) + length(repeated)
        rown    =  length(Tables.rows(data))
        #
        q       = Vector{Int}(undef, alleffl)
        t       = Vector{Int}(undef, alleffl)
        tr      = Vector{UnitRange{Int}}(undef, alleffl)
        schema  = Vector{Union{AbstractTerm, Tuple}}(undef, alleffl)
        z       = Matrix{Float64}(undef, rown, 0)
        #subjz   = Vector{BitMatrix}(undef, alleffl)
        dicts   = Vector{Dict}(undef, alleffl)
        # unit range z column range for each random effect
        zrndur  = Vector{UnitRange{Int}}(undef, length(random))
        # Number of random effects
        rn      = length(random)
        #
        rtn     = 0 
        # Number of repeated effects
        rpn     = length(repeated)
        # Z Matrix for repeated effect
        # rz      = Vector{Matrix{Float64}}(undef, rpn)
        # 
        #Theta parameter type
        ct      = Vector{Symbol}(undef, 0)
        # emap
        emap    = Vector{Int}(undef, 0)
        # Names
        rcnames = Vector{String}(undef, 0)
        #
        sn      = zeros(Int, alleffl)
        if rn > 1
            @inbounds for i = 2:rn
                if !random[i].covtype.z error("One of the random effect have zero type!") end
            end
        end
        # RANDOM EFFECTS
            @inbounds for i = 1:rn
                if length(random[i].coding) == 0
                    fill_coding_dict!(random[i].model, random[i].coding, data)
                end
                if isa(random[i].model, ConstantTerm) # if only ConstantTerm in the model - data_ - first is collumn (responce)
                    data_     = data[[first(keys(data))]] 
                else
                    data_     = data[StatsModels.termvars(random[i].model)] # only collumns for model
                end
                if isa(random[i].covtype.s, ZERO)
                    schema[i] = InterceptTerm{false}()
                    zsize     = 0
                else
                    schema[i] = apply_schema(random[i].model, StatsModels.schema(data_, random[i].coding))
                    ztemp     = modelcols(MatrixTerm(schema[i]), data_)
                    z         = hcat(z, ztemp)
                    zsize     = size(ztemp, 2)
                end
                
                q[i]      = zsize
                csp       = covstrparam(random[i].covtype.s, q[i])
                t[i]      = sum(csp)
                
                fillur!(zrndur, i, q)
                fillur!(tr, i, t)
                symbs       = StatsModels.termvars(random[i].subj)
                if length(symbs) > 0
                    cdata     = tabcols(data, symbs) 
                    dicts[i]  = Dict{Tuple{eltype.(cdata)...}, Vector{Int}}()
                    indsdict!(dicts[i], cdata)
                else
                    dicts[i]  = Dict(1 => collect(1:rown)) #changed to range
                end

                sn[i]     = length(dicts[i])
                updatenametype!(ct, rcnames, csp, schema[i], random[i].covtype.s)
                append!(emap, fill(i, t[i]))
                rtn += t[i]
            end
        
        rz_      = Vector{Matrix}(undef, rpn)
        # REPEATED EFFECTS
        for i = 1:length(repeated)

            if isa(repeated[i].covtype.s, ACOV_) && i == 1
                @warn "ACOV at first position is meaningless: base covariance is not yet computed."
            end

            if length(repeated[i].coding) == 0
                fill_coding_dict!(repeated[i].model, repeated[i].coding, data)
            end
            if isa(repeated[i].model, ConstantTerm) # if only ConstantTerm in the model - data_ - first is collumn (responce)
                data_     = data[[first(keys(data))]] 
            else
                data_     = data[StatsModels.termvars(repeated[i].model)] # only collumns for model
            end
            
            schema[rn + i] = apply_schema(repeated[i].model, StatsModels.schema(data_, repeated[i].coding))
            rz_[i]       = modelcols(MatrixTerm(schema[rn+i]), data_)
            symbs        = StatsModels.termvars(repeated[i].subj)
            if length(symbs) > 0
                cdata    = tabcols(data, symbs) 
                dicts[rn + i]  = Dict{Tuple{eltype.(cdata)...}, Vector{Int}}()
                indsdict!(dicts[rn + i], cdata)
            else
                dicts[rn+i]  = Dict(1 => collect(1:rown)) #changed to range
            end
            # If UN structure used all repeated levels should be unique within one subject, otherwise results can be meaningless!
            wflag = true
            if isa(repeated[i].covtype.s, UN_)
                for (k,v) in dicts[rn+i]
                    sv = view(rz_[i], v, :)
                    for j = 1:size(sv, 2)
                        if sum(view(sv, :, j)) > 1 && wflag
                            wflag = false
                            @warn "If UN structure used for repeated effect all levels should be unique within one subject, otherwise results can be meaningless!"
                        end
                    end
                end
            end

            sn[rn + i]   = length(dicts[rn+i])
            q[rn + i]    = size(rz_[i], 2)
            csp          = covstrparam(repeated[i].covtype.s, q[rn+i])
            t[rn + i]    = sum(csp)
            tr[rn + i]   = UnitRange(sum(t[1:rn+i-1]) + 1, sum(t[1:rn+i-1]) + t[rn+i])
            updatenametype!(ct, rcnames, csp, schema[rn+i], repeated[i].covtype.s)
            # emap
            append!(emap, fill(rn+i, t[rn+i]))
        end
        T2  = typejoin(eltype.(rz_)...)
        rz  = Vector{Matrix{T2}}(undef, rpn)
        rz .= rz_
        # Theta length
        tl  = sum(t)
        ########################################################################
        cross = Int[]
        if random[1].covtype.z
            append!(cross, 1:rn)                       # все случайные эффекты
        end
        repn = Int[]
        for i = 1:rpn
            if isa(repeated[i].covtype.s, SI_) || isa(repeated[i].covtype.s, DIAG_)
                push!(repn, i)                         # диагональные не связывают
            else
                push!(cross, rn + i)
            end
        end
        if isempty(cross)
        # связывающих эффектов нет: блокируем по первому repeated-эффекту
            subjblockdict = _canonical_blocks(dicts[rn + 1])
        else
            subjblockdict = crossdicts(dicts, cross, rown)   # один проход DSU
        end

        # диагональные repeated-эффекты наследуют блоки
        for i in repn
            dicts[rn + i] = subjblockdict
        end

        #=
        if random[1].covtype.z  # if first random effect not null
            subjblockdict = dicts[1]
            if length(dicts) > 2 # if more than 2 random effects
                for i = 2:length(dicts)-1
                    subjblockdict = subjcrossdicts(subjblockdict, dicts[i])
                end
            end
        else
            subjblockdict = nothing
        end
        repn = Int[]
        for i = 1:length(repeated)
            if isnothing(subjblockdict)
                subjblockdict = dicts[rn+i]
            elseif !(isa(repeated[i].covtype.s, SI_) || isa(repeated[i].covtype.s, DIAG_)) # if repeated effect have non-diagonal structure
                subjblockdict = subjcrossdicts(subjblockdict, dicts[rn+i]) # make dict for non SI DIAG repeated effects 
            else
                push!(repn, i) # just collect ind of SI DIAG repeated effects 
            end
        end
        for i in repn # make SI DIAG repeated effects dict - subjblockdict
            dicts[rn+i] = subjblockdict
        end
        =#
        blocks = [subjblockdict[b] for b in 1:length(subjblockdict)]
        maxn   = maximum(length, blocks)

        esb    = make_effect_subject_block(dicts, blocks, alleffl, rown)
        #######################################################################
        # Postprocessing
        # Modify repeated effect covariance type for some types
        # Maybe it will be removed
        for r in repeated 
            applycovschema!(r.covtype.s, blocks)
        end
        #######################################################################
        new{eltype(z), T2}(random, repeated, schema, rcnames, blocks, rn, rtn, rpn, z, esb, zrndur, rz, q, t, tr, tl, ct, emap, sn, maxn)
    end
end
###############################################################################
function fillur!(ur, i, v)
    if i > 1
        ur[i]   = UnitRange(sum(v[1:i-1]) + 1, sum(v[1:i-1]) + v[i])
    else
        if v[1] > 0
            ur[1]   = UnitRange(1, v[1])
        else
            ur[1]   = UnitRange(0, 0)
        end
    end
end
################################################################################
function updatenametype!(ct, rcnames, csp, schema, s)
    append!(ct, fill!(Vector{Symbol}(undef, csp[1]), :var))
    append!(ct, fill!(Vector{Symbol}(undef, csp[2]), :rho))
    if length(csp) == 3 append!(ct, fill!(Vector{Symbol}(undef, csp[3]), :theta)) end
    append!(rcnames, rcoefnames(schema, sum(csp), s))
end

################################################################################
#                            CONTRAST CODING
################################################################################

function fill_coding_dict!(t::T, d::Dict, data) where T <: Union{ConstantTerm, InterceptTerm}
    return d
end
function fill_coding_dict!(t::T, d::Dict, data) where T <: FunctionTerm
    if t.f === +
        for i in t.args
            fill_coding_dict!(i, d, data)
        end
    end
    return d
end
function fill_coding_dict!(t::T, d::Dict, data) where T <: Term
    if typeof(Tables.getcolumn(data, t.sym)) <: AbstractCategoricalVector || !(typeof(Tables.getcolumn(data, t.sym)) <: AbstractVector{V} where V <: Real)
        d[t.sym] = StatsModels.FullDummyCoding()
    end
    return d
end
#=
function fill_coding_dict!(t::T, d::Dict, data) where T <: InteractionTerm
    for i in t.terms
        if typeof(Tables.getcolumn(data, i.sym))  <: AbstractCategoricalVector || !(typeof(Tables.getcolumn(data, i.sym)) <: AbstractVector{V} where V <: Real)
            d[i.sym] = StatsModels.FullDummyCoding()
        end
    end
    d
end
=#
function fill_coding_dict_ct!(t, d, data)
    for i in t
        if isa(i, Term)
            if typeof(Tables.getcolumn(data, i.sym)) <: AbstractCategoricalVector || !(typeof(Tables.getcolumn(data, i.sym)) <: AbstractVector{V} where V <: Real)
                d[i.sym] = StatsModels.FullDummyCoding()
            end
        else
            fill_coding_dict!(i, d, data)
        end
    end
    return d
end
#=
function fill_coding_dict!(t::T, d::Dict, data) where T <: Tuple{Vararg{AbstractTerm}}
    fill_coding_dict_ct!(t, d, data)
end
=#
function fill_coding_dict!(t::T, d::Dict, data) where T <: CType
    fill_coding_dict_ct!(t.args, d, data)
end
#=
function fill_coding_dict!(t::T, d::Dict, data) where T <: FunctionTerm{typeof(&), Vector{Term}}
    for i in t.args
        if isa(i, Term)
            if typeof(Tables.getcolumn(data, i.sym)) <: AbstractCategoricalVector || !(typeof(Tables.getcolumn(data, i.sym)) <: AbstractVector{V} where V <: Real)
                d[i.sym] = StatsModels.FullDummyCoding()
            end
        else
            fill_coding_dict!(i, d, data)
        end
    end
    d
end
function fill_coding_dict!(t::T, d::Dict, data) where T <: FunctionTerm{typeof(+), Vector{Term}}
    for i in t.args
        if isa(i, Term)
            if typeof(Tables.getcolumn(data, i.sym)) <: AbstractCategoricalVector || !(typeof(Tables.getcolumn(data, i.sym)) <: AbstractVector{V} where V <: Real)
                d[i.sym] = StatsModels.FullDummyCoding()
            end
        else
            fill_coding_dict!(i, d, data)
        end
    end
    d
end
=#
################################################################################
# SHOW
################################################################################
function Base.show(io::IO, e::VarEffect)
    println(io, "  Formula: ", e.formula)
    println(io, "  Effect model: ", e.model)
    println(io, "  Subject model: ", e.subj)
    println(io, "  Type: ", e.covtype.s)
    print(io, "  User coding:")
    if length(e.coding) > 0
        for (k, v) in e.coding
            print(io, " $(k) => $(v);")
        end
    else
        print(io, " No")
    end
end

function Base.show(io::IO, cs::CovStructure)
    println(io, "Covariance Structure:")
    for i = 1:length(cs.random)
        println(io, "Random $(i):", cs.random[i])
    end
    for i = 1:length(cs.repeated)
        println(io, "Repeated $(i): ", cs.repeated[i])
    end
    println(io, "Random effect range in complex Z: ", cs.zrndur)
    println(io, "Random coef. in θ: ", cs.rtn)
    println(io, "Range of each parameters in θ vector: ", cs.tr)
    println(io, "Size of Z: ", cs.q)
    println(io, "Parameter number for each effect: ", cs.t)
    println(io, "Theta length:", cs.tl)
end
