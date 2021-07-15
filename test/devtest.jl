using DataFrames, CSV, StatsModels, LinearAlgebra, ForwardDiff, BenchmarkTools, ForwardDiff, Optim, Distributions
using NLopt
using SnoopCompile
using LineSearches
path    = dirname(@__FILE__)
cd(path)
df      = CSV.File(path*"/csv/df0.csv") |> DataFrame

θ = zeros(10, 10)
A = rand(10, 10)
B = rand(10, 10)

@benchmark Metida.mulαβαtinc!(θ, A, B)

θ2 = zeros(10)
b = rand(10)
@benchmark Metida.mulαtβinc!(θ2, A, b)

mx = zeros(10,10)
θ3 = [1.2, 3.4]
rz = rand(10, 2)
Metida.rmatp_diag!(mx, θ3, rz, 0)

Metida.rmatp_si!(mx, θ3, rz, 0)

θ4 = [1.2, 3.4, 0.2]
Metida.rmatp_csh!(mx, θ4, rz, 0)

function mult(rx::AbstractMatrix{T}, θ) where T
    vec = Vector{T}(undef, size(rz, 1))
    for r ∈ axes(rz, 1)
        for i ∈ axes(rz, 2)
            vec[r] += rz[r, i] * θ[i]
        end
    end
    vec
end
