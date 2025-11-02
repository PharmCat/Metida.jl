# Metida
# Copyright © 2019-2020 Vladimir Arnautov aka PharmCat <mail@pharmcat.net>

using Metida

include("test.jl")

import Aqua

@testset "Aqua                                                       " begin
    Aqua.test_all(Metida, piracies=false) # fix piracies with MetidaBase 
end
