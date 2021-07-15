using Metida
using SnoopCompile
path    = dirname(@__FILE__)
cd(path)

inf_timing = @snoopi tmin=0.00001 include("test.jl")
pc = SnoopCompile.parcel(inf_timing)
SnoopCompile.write("precompile", pc)


SnoopCompile.@snoopc "$path/precompile/metida_compiles.log" begin
    using Metida, Pkg
    include(joinpath(dirname(dirname(pathof(Metida))), "test", "test.jl"))
end
data = SnoopCompile.read("$path/precompile/metida_compiles.log")
pc = SnoopCompile.parcel(reverse!(data[2]))
SnoopCompile.write("$path/precompile", pc)


using SnoopCompileCore
invalidations = @snoopr begin
    using Metida
end
using SnoopCompile
ui = uinvalidated(invalidations)
trees = invalidation_trees(invalidations)
ftrees = filtermod(Metida, trees)
