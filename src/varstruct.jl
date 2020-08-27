macro covstr(ex)
    return :(@formula(nothing ~ $ex).rhs)
end


struct VarianceComponents <: AbstractCovarianceType
end
VC = VarianceComponents

struct ScaledIdentity <: AbstractCovarianceType
end
SI = ScaledIdentity

#schema(df6)
#rschema = apply_schema(Term(formulation), schema(df, Dict(formulation => StatsModels.FullDummyCoding())))
#apply_schema(Term(formulation), shema1)
#Z   = modelcols(rschema, df)
#reduce(hcat, Z)

struct VarEffect
    model
    covtype::AbstractCovarianceType
    function VarEffect(model, covtype)
        new(model, covtype)
    end
    function VarEffect(model)
        new(model, VarianceComponents())
    end
    function VarEffect()
        new(nothing, ScaledIdentity())
    end
end

struct CovStructure <: AbstractCovarianceStructure
    random
    repeated::VarEffect
end
