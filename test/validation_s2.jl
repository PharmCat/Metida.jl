

df        = CSV.File(joinpath(path, "csv", "Penicillin.csv"); types = [String, Float64, String, String]) |> DataFrame

@testset "  Model 7: Penicillin.csv parameters                       " begin
    lmm = Metida.LMM(@formula(diameter ~ 1), df;
    random = [Metida.VarEffect(@covstr(1|plate), Metida.SI), Metida.VarEffect(@covstr(1|sample), Metida.SI)]
    )
    Metida.fit!(lmm)
    @test Metida.coef(lmm)[1]     ≈ 22.9722 atol = 1E-4
    @test Metida.stderror(lmm)[1] ≈ 0.808573 atol = 1E-4
    @test Metida.theta(lmm)[1]^2 ≈ 0.716908  atol = 1E-4
    @test Metida.theta(lmm)[2]^2 ≈ 3.73092 atol = 1E-4
    @test Metida.theta(lmm)[3]^2 ≈ 0.302415 atol = 1E-4
end
