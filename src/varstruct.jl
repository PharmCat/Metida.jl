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
    eff, subj = term.args_parsed
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
    subj::Union{Tuple{Vararg{AbstractTerm}}, AbstractTerm}
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
struct CovStructure{T} <: AbstractCovarianceStructure
    # Random effects
    random::Vector{VarEffect}
    # Repearted effects
    repeated::VarEffect
    schema::Vector{Union{Tuple, AbstractTerm}}
    rcnames::Vector{String}
    # blocks for vcov matrix / variance blocking factor (subject)
    vcovblock::Vector{Vector{Int}}
    # number of random effect
    rn::Int
    # Z matrix
    z::Matrix{T}
    #subjz::Vector{BitArray{2}}
    # Blocks for each blocking subject, each effect, each effect subject sblock[block][rand eff][subj]
    sblock::Vector{Vector{Vector{Vector{Int}}}}
    #unit range z column range for each random effect
    zrndur::Vector{UnitRange{Int}}
    # repeated effect parametrization matrix
    rz::Matrix{T}
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
    # Nubber of subjects in each effect
    sn::Vector{Int}
    # Maximum number per block
    maxn::Int
    #--
    function CovStructure(random, repeated, data)
        alleffl =  length(random) + 1
        rown    =  length(Tables.rows(data))
        #
        q       = Vector{Int}(undef, alleffl)
        t       = Vector{Int}(undef, alleffl)
        tr      = Vector{UnitRange{Int}}(undef, alleffl)
        schema  = Vector{Union{AbstractTerm, Tuple}}(undef, alleffl)
        z       = Matrix{Float64}(undef, rown, 0)
        subjz   = Vector{BitMatrix}(undef, alleffl)
        zrndur  = Vector{UnitRange{Int}}(undef, alleffl - 1)
        rz      = Matrix{Float64}(undef, rown, 0)
        rn      = length(random)
        #Theta parameter type
        ct  = Vector{Symbol}(undef, 0)
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
            schema[i] = apply_schema(random[i].model, StatsModels.schema(data, random[i].coding))
            ztemp     = modelcols(MatrixTerm(schema[i]), data)
            q[i]      = size(ztemp, 2)
            csp       = covstrparam(random[i].covtype.s, q[i])
            t[i]      = sum(csp)
            z         = hcat(z, ztemp)
            fillur!(zrndur, i, q)
            fillur!(tr, i, t)

            subjz[i]  = convert(BitMatrix, modelcols(MatrixTerm(apply_schema(random[i].subj, StatsModels.schema(data, fulldummycodingdict(random[i].subj)))), data))
            sn[i] = size(subjz[i], 2)
            updatenametype!(ct, rcnames, csp, schema[i], random[i].covtype.s)
        end
        # REPEATED EFFECTS
        if length(repeated.coding) == 0
            fill_coding_dict!(repeated.model, repeated.coding, data)
        end

        schema[end] = apply_schema(repeated.model, StatsModels.schema(data, repeated.coding))
        rz          = modelcols(MatrixTerm(schema[end]), data)

        subjz[end]  = convert(BitMatrix, modelcols(MatrixTerm(apply_schema(repeated.subj, StatsModels.schema(data, fulldummycodingdict(repeated.subj)))), data))
        sn[end] = size(subjz[end], 2)
        q[end]      = size(rz, 2)
        csp         = covstrparam(repeated.covtype.s, q[end])
        t[end]      = sum(csp)
        tr[end]     = UnitRange(sum(t[1:end-1]) + 1, sum(t[1:end-1]) + t[end])
        updatenametype!(ct, rcnames, csp, schema[end], repeated.covtype.s)
        #Theta length
        tl  = sum(t)
        ########################################################################
        if random[1].covtype.z
            subjblockmat = subjz[1]
            if length(subjz) > 2
                for i = 2:length(subjz)-1
                    subjblockmat = noncrossmodelmatrix(subjblockmat, subjz[2])
                end
            end
            if !(isa(repeated.covtype.s, SI_) || isa(repeated.covtype.s, DIAG_))
                subjblockmat = noncrossmodelmatrix(subjblockmat, subjz[end])
            end
        else
            subjblockmat = subjz[end]
        end
        blocks = makeblocks(subjblockmat) #vcovblock
        sblock = Vector{Vector{Vector{Vector{UInt32}}}}(undef, length(blocks))
        ########################################################################
        @inbounds for i = 1:length(blocks)
            sblock[i] = Vector{Vector{Vector{UInt32}}}(undef, alleffl)
            @inbounds for s = 1:alleffl
                sblock[i][s] = Vector{Vector{UInt32}}(undef, 0)
                @inbounds for col in eachcol(view(subjz[s], blocks[i], :))
                    if any(col) push!(sblock[i][s], findall(col)) end
                end
            end
        end
        #
        maxn = 0
        for i in blocks
            lvcb = length(i)
            if lvcb > maxn maxn = lvcb end
        end
        new{eltype(z)}(random, repeated, schema, rcnames, blocks, rn, z, sblock, zrndur, rz, q, t, tr, tl, ct, sn, maxn)
    end
end
################################################################################
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
function makeblocks(subjz)
    blocks = Vector{Vector{UInt32}}(undef, 0)
    for i = 1:size(subjz, 2)
        b = findall(x->!iszero(x), view(subjz, :, i))
        if length(b) > 0 push!(blocks, b) end
    end
    blocks
end
################################################################################
function noncrossmodelmatrix(mx::AbstractArray, my::AbstractArray)
    size(mx, 2) > size(my, 2) ?  (mat = mx' * my; a = mx) : (mat = my' * mx; a = my)
    #mat = mat * mat'
    T = eltype(mat)
    @inbounds for n = 1:size(mat, 2)-1
        fr = findfirst(x->!iszero(x), view(mat, n, :))
        if !isnothing(fr)
            @inbounds for m = fr:size(mat, 1)
                if !iszero(mat[m, n])
                    fc = findfirst(x->!iszero(x), view(mat, m, n+1:size(mat, 2)))
                    if !isnothing(fc)
                        @inbounds for c = n+fc:size(mat, 2)
                            if !iszero(mat[m, c])
                                #view(mat, :, n) .+= view(mat, :, c)
                                A = view(mat, :, n)
                                broadcast!(+, A, A, view(mat, :, c))
                                fill!(view(mat, :, c), zero(T))
                            end
                        end
                    end
                end
            end
        end
    end
    cols = Vector{Int}(undef, 0)
    @inbounds for i = 1:size(mat, 2)
        if !iszero(sum(view(mat,:, i)))
            push!(cols, i)
        end
    end
    res = replace(x -> iszero(x) ?  0 : 1, view(mat, :, cols))
    a * res
end
################################################################################
#                            CONTRAST CODING
################################################################################

function fill_coding_dict!(t::T, d::Dict, data) where T <: Union{ConstantTerm, InterceptTerm, FunctionTerm}
end
function fill_coding_dict!(t::T, d::Dict, data) where T <: Term
    if typeof(Tables.getcolumn(data, t.sym)) <: CategoricalArray || !(typeof(Tables.getcolumn(data, t.sym)) <: Vector{T} where T <: Real)
        d[t.sym] = StatsModels.FullDummyCoding()
    end
end
#function fill_coding_dict!(t::T, d::Dict, data) where T <: CategoricalTerm
#    if typeof(data[!, t.sym])  <: CategoricalArray
#        d[t.sym] = StatsModels.FullDummyCoding()
#    end
#end
function fill_coding_dict!(t::T, d::Dict, data) where T <: InteractionTerm
    for i in t.terms
        if typeof(Tables.getcolumn(data, i.sym))  <: CategoricalArray || !(typeof(Tables.getcolumn(data, i.sym)) <: Vector{T} where T <: Real)
            d[i.sym] = StatsModels.FullDummyCoding()
        end
    end
end
function fill_coding_dict!(t::T, d::Dict, data) where T <: Tuple{Vararg{AbstractTerm}}
    for i in t
        if isa(i, Term)
            if typeof(Tables.getcolumn(data, i.sym)) <: CategoricalArray || !(typeof(Tables.getcolumn(data, i.sym)) <: Vector{T} where T <: Real)
                d[i.sym] = StatsModels.FullDummyCoding()
            end
        else
            fill_coding_dict!(i, d, data)
        end
    end
end
function fulldummycodingdict(t::InteractionTerm)
    d = Dict{Symbol, AbstractContrasts}()
    for i in t.terms
        d[i.sym] = StatsModels.FullDummyCoding()
    end
    d
end
function fulldummycodingdict(t::T) where T <: Union{CategoricalTerm, Term}
    d = Dict{Symbol, AbstractContrasts}()
    d[t.sym] = StatsModels.FullDummyCoding()
    d
end
function fulldummycodingdict(t::T) where T <: Union{ConstantTerm, InterceptTerm}
    d = Dict{Symbol, AbstractContrasts}()
    d
end
function fulldummycodingdict(t::Tuple{Vararg{AbstractTerm}})
    d = Dict{Symbol, AbstractContrasts}()
    for i in t
        merge!(d, fulldummycodingdict(i))
    end
    d
end

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
    println(io, "Repeated: ", cs.repeated)
    println(io, "Random effect range in complex Z: ", cs.zrndur)
    println(io, "Size of Z: ", cs.q)
    println(io, "Parameter number for each effect: ", cs.t)
    println(io, "Theta length:", cs.tl)
end
