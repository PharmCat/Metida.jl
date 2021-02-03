# Metida

#Simple dataset
df0         = CSV.File(path*"/csv/df0.csv"; types = [String, String, String, String, Float64, Float64]) |> DataFrame

ftdf         = CSV.File(path*"/csv/1fptime.csv"; types = [String, String, Float64, Float64]) |> DataFrame
