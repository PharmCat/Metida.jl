struct LMMformula
    formula
    random
    repeated
end

"""
    @lmmformula(formula, args...)

Macro for made formula with variance-covariance structure representation. `@lmmformula`
could be used for shorter `LMM` construction.

Example:

```
lmm = Metida.LMM(@lmmformula(var~sequence+period+formulation,
random = formulation|subject:CSH,
repeated = formulation|subject:DIAG),
df0)
```

equal to:

```
lmm = LMM(@formula(var~sequence+period+formulation), df0;
random = Metida.VarEffect(@covstr(formulation|subject), CSH),
repeated = Metida.VarEffect(@covstr(formulation|subject), DIAG),
)
```

`@lmmformula` have 3 components - 1'st is a formula for fixed effect, it defined
like in `StstsModels` (1st argument just provided to `@formula` macro). Other arguments
should be defined like keywords. `repeated` keyword define repeated effect part,
`random` define random effect part. You can use several random factors as in example bellow:

```
lmm = Metida.LMM(Metida.@lmmformula(var~sequence+period+formulation,
random = formulation|subject:CSH,
random = 1|subject:DIAG,
repeated = formulation|subject:DIAG),
df0)
```

`random` or `repeated` structure made by template:

`effect formula` | `blocking factor` [/ `nested factor`] [: `covariance structure`]

`|` - devide effect formula form blocking factor definition (necessarily),
`/` and `:` modificator are optional. `/` work like in MixedModels or in RegressionFormulae -
expand factor `f|a/b` to `f|a` + `f|a&b`. It can't be used in repeated effect defenition.

`:` - covariance structure defined right after `:` (SI, DIAG, CS, CSH, ets...),
if `:` not used then SI used for this effect.

Terms like `a+b` or `a*b` shuould not be used as a blocking factors.

"""
macro lmmformula(formula, args...)
    f = eval(:(@formula($formula)))
    ranfac = Metida.VarEffect[]
    repeff = nothing
    for ex in args
        if ex.head != :(=) error("Error = ") end
        a, e = ex.args
        ear = e.args
        # Check if structure defined, else SI used
        if ear[1] == :(:)
            ct = eval(ear[3])
            e = ear[2]
        else
            ct = SI
        end
        ear = e.args
        if a == :random
            if ear[1] == :(|)
                # For simple random factor
                if isa(ear[3], Symbol)
                    mcs = eval(:(@covstr($e)))
                    push!(ranfac, VarEffect(mcs, ct))
                # For nested factors defined with "/"
                elseif length(ear[3].args) == 3 && ear[3].args[1] == :(/)
                    o = ear[3].args[2]
                    mcs = eval(:(@covstr($(ear[2]) | $o)))
                    push!(ranfac, VarEffect(mcs, ct))
                    u = ear[3].args[3]
                    t = Expr(:call, :&, o, u)
                    mcs = eval(:(@covstr($(ear[2]) | $t)))
                    push!(ranfac, VarEffect(mcs, ct))
                # Other cases
                else
                    mcs = eval(:(@covstr($e)))
                    push!(ranfac, VarEffect(mcs, ct))
                end
            else
                error("Random term should include `|` operator in syntax")
            end

        elseif a == :repeated
            if ear[1] == :(|)
                if isa(ear[3], Symbol)

                    mcs = eval(:(@covstr($e)))
                    repeff = VarEffect(mcs, ct)

                elseif length(ear[3].args) == 3 && ear[3].args[1] == :(/)

                    error("Repeated effect can't include `/`")

                else
                    mcs = eval(:(@covstr($e)))
                    repeff = VarEffect(mcs, ct)
                end
            else
                error("Random term should include `|` operator in syntax")
            end

        else
            error("Unknown type")
        end
    end
    if length(ranfac) == 0 ranfac = nothing end
    return LMMformula(f, ranfac, repeff)
end
