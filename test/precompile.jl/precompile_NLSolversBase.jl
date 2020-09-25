function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    isdefined(NLSolversBase, Symbol("#42#48")) && Base.precompile(Tuple{getfield(NLSolversBase, Symbol("#42#48")),Array{Float64,1},Array{Float64,1}})
    isdefined(NLSolversBase, Symbol("#43#49")) && Base.precompile(Tuple{getfield(NLSolversBase, Symbol("#43#49")),Array{Float64,2},Array{Float64,1}})
end
