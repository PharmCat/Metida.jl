################################################################################
#                       ПРОГРАММА ВАЛИДАЦИОННЫХ ТЕСТОВ
################################################################################

include(joinpath(dirname(@__FILE__), "validation_init.jl"))


println("\n", "="^78, "\nКАТЕГОРИЯ 1: публичные наборы, MixedModels.jl и SPSS\n", "="^78)
include(joinpath(path, "validation_mm.jl"))

println("\n", "="^78, "\nКАТЕГОРИЯ 2: биоэквивалентность, SPSS\n", "="^78)
include(joinpath(path, "validation_be.jl"))

println("\n", "="^78, "\nКАТЕГОРИЯ 3: параметры ковариации, SPSS\n", "="^78)
include(joinpath(path, "validation_spss.jl"))
