# Metida

#Simple dataset
df0         = CSV.File(path*"/csv/df0.csv") |> DataFrame
categorical!(df0, :subject);
categorical!(df0, :period);
categorical!(df0, :sequence);
categorical!(df0, :formulation);
