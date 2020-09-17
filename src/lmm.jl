#lmm.jl

#Metida.LMM(@formula(var ~ formulation + period), df6)

struct LMM{T} <: MetidaModel
    model::FormulaTerm
    mf::ModelFrame
    mm::ModelMatrix
    covstr::CovStructure

    function LMM(model, data; contrasts=Dict{Symbol,Any}(), subject = nothing,  random = nothing, repeated = nothing)
        mf = ModelFrame(model, data; contrasts = contrasts)
        mm = ModelMatrix(mf)
        if random === nothing
            random = VarEffect()
        end
        if repeated === nothing
            repeated = VarEffect()
        end
        if !isa(random, Vector) random = [random] end
        covstr = CovStructure(random, repeated, data)

        #z      = get_z_matrix(data, covstr)

        #terms  = get_term_vec(covstr)
        #tdict  = Dict{Symbol, AbstractContrasts}()
        #filltdict(terms, tdict)
        #rschema = apply_schema(terms,
        #    schema(data, tdict)
        #    )
        #Z   = modelcols(rschema, data)

        #new{eltype(mm.m)}(model, mf, mm, covstr, Z)
        new{eltype(mm.m)}(model, mf, mm, covstr)
    end
end


function Base.show(io::IO, lmm::LMM)
    println(io, "Linear Mixed Model: ", lmm.model)
end
