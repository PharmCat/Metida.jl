# Metida

#Simple dataset
df0         = CSV.File(path*"/csv/df0.csv"; types = [String, String, String, String, Float64, Float64, Float64]) |> DataFrame

df0m         = CSV.File(path*"/csv/df0miss.csv"; types = [String, String, String, String, Float64, Float64]) |> DataFrame


df1         = CSV.File(path*"/csv/df1.csv"; types = [String, String, String, String, Float64, Float64]) |> DataFrame

ftdf         = CSV.File(path*"/csv/1fptime.csv"; types = [String, String, Float64, Float64]) |> DataFrame

ftdf2        = CSV.File(path*"/csv/1freparma.csv"; types = [String, String, Float64, Float64]) |> DataFrame

ftdf3        = CSV.File(path*"/csv/ftdf3.csv"; types =
[String,  Float64, Float64, String, String, String, String, String, Float64]) |> DataFrame

spatdf       = CSV.File(path*"/csv/spatialdata.csv"; types = [Int, Int, String, Float64, Float64, Float64, Float64, Float64]) |> DataFrame


dfrdsfda        = CSV.File(joinpath(path, "csv", "berds", "rds1.csv"), types = Dict(:PK => Float64, :subject => String, :period => String, :sequence => String, :treatment => String )) |> DataFrame
dropmissing!(dfrdsfda)
dfrdsfda.lnpk = log.(dfrdsfda.PK)
