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
    #v = d[t.sym]
    #reshape(v, length(v), 1)  
    d[t.sym]
end

################################################################################
#                     @covstr macro
################################################################################

"""
    @covstr(ex)

Macros for random/repeated effect model.

# Example

```julia
@covstr(factor|subject)
```
"""
macro covstr(ex)
    return :(@formula(nothing ~ $ex).rhs)
end
function modelparse(term::FunctionTerm{typeof(|)})
    eff, subj = term.args
    if !isa(subj, AbstractTerm) || isa(subj, FunctionTerm{typeof(*), Vector{Term}}) throw(FormulaException("Subject term type not <: AbstractTerm. Use `term` or `interaction term` only. Maybe you are using something like this: `@covstr(factor|term1*term2)` or `@covstr(factor|(term1+term2))`. Use only `@covstr(factor|term)` or `@covstr(factor|term1&term2)`.")) end
    eff, subj
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

    Categorical factors are coded with `FullDummyCoding()` by default, use `coding` for other contrast codeing.

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
    v
end

tabcols(data, symbs) = Tuple(Tables.getcolumn(Tables.columns(data), x) for x in symbs)

struct EffectSubjectBlock
    sblock::Matrix{Vector{Tuple{Vector{Int}, Int}}}
    snames::Vector
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
    v
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
        #if random[1].covtype.z #IF NOT ZERO
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
                    cdata     = tabcols(data, symbs) # Tuple(Tables.getcolumn(Tables.columns(data_), x) for x in symbs)
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

            if length(repeated[i].coding) == 0
                fill_coding_dict!(repeated[i].model, repeated[i].coding, data)
            end
            if isa(repeated[i].model, ConstantTerm) # if only ConstantTerm in the model - data_ - first is collumn (responce)
                data_     = data[[first(keys(data))]] 
            else
                data_     = data[StatsModels.termvars(repeated[i].model)] # only collumns for model
            end
            
            schema[rn + i] = apply_schema(repeated[i].model, StatsModels.schema(data_, repeated[i].coding))
            #rz_[i]       = reduce(hcat, modelcols(schema[rn+i], data))
            rz_[i]       = modelcols(MatrixTerm(schema[rn+i]), data_)
            symbs        = StatsModels.termvars(repeated[i].subj)
            if length(symbs) > 0
                cdata    = tabcols(data, symbs) # Tuple(Tables.getcolumn(Tables.columns(data), x) for x in symbs)
                dicts[rn + i]  = Dict{Tuple{eltype.(cdata)...}, Vector{Int}}()
                indsdict!(dicts[rn + i], cdata)
            else
                dicts[rn+i]  = Dict(1 => collect(1:rown)) #changed to range
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
        #if any(x-> 1 in keys(x), dicts[1:end-1])
        #    blocks = [first(dicts)[1]]
        #else
            if random[1].covtype.z  # if first random effect not null
                subjblockdict = dicts[1]
                if length(dicts) > 2 # if more than 2 random effects
                    for i = 2:length(dicts)-1
                        subjblockdict = sabjcrossdicts(subjblockdict, dicts[i])
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
                    subjblockdict = sabjcrossdicts(subjblockdict, dicts[rn+i]) # make dict for non SI DIAG repeated effects 
                else
                    push!(repn, i) # just collect ind of SI DIAG repeated effects 
                end
            end
            for i in repn # make SI DIAG repeated effects dict - subjblockdict
                dicts[rn+i] = subjblockdict
            end


            blocks  = collect(values(subjblockdict))
        #end

        sblock = Matrix{Vector{Tuple{Vector{Int}, Int}}}(undef, length(blocks), alleffl)
        nblock = []
        #######################################################################
        #######################################################################
        nli = 1
        @inbounds for i = 1:length(blocks) # i - block number
            @inbounds for s = 1:alleffl # s - effect number
                tempv = Vector{Tuple{Vector{Int}, Int}}(undef, 0)
                for (k, v) in dicts[s]
                    fa = findall(x-> x in v, blocks[i]) # Try to optimize it
                    if length(fa) > 0 
                        push!(tempv, (fa, nli))
                        push!(nblock, k)
                        nli += 1
                    end
                end
                sblock[i, s] = tempv
            end
        end
        #
        maxn = 0
        for i in blocks
            lvcb = length(i)
            if lvcb > maxn maxn = lvcb end
        end
        esb = EffectSubjectBlock(sblock, nblock)
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

function fill_coding_dict!(t::T, d::Dict, data) where T <: Union{ConstantTerm, InterceptTerm, FunctionTerm}
    d
end
function fill_coding_dict!(t::T, d::Dict, data) where T <: Term
    if typeof(Tables.getcolumn(data, t.sym)) <: AbstractCategoricalVector || !(typeof(Tables.getcolumn(data, t.sym)) <: AbstractVector{V} where V <: Real)
        d[t.sym] = StatsModels.FullDummyCoding()
    end
    d
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
    d
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
