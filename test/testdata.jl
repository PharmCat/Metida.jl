# Metida

#Simple dataset
df0         = CSV.File(path*"/csv/df0.csv"; types = [String, String, String, String, Float64, Float64]) |> DataFrame

ftdf         = CSV.File(path*"/csv/1fptime.csv"; types = [String, String, Float64, Float64]) |> DataFrame

ftdf2        = CSV.File(path*"/csv/1freparma.csv"; types = [String, String, Float64, Float64]) |> DataFrame

ftdf3        = CSV.File(path*"/csv/2f2rand.csv"; types =
[String,  Float64, Float64, String, String, String, String, String]) |> DataFrame
