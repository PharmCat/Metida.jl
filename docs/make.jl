using Documenter, Metida

makedocs(
    modules = [Metida],
    sitename = "Metida Documentation",
    authors = "Vladimir Arnautov",
    linkcheck = false,
    doctest = false,
    pages = [
        "Home" => "index.md",
        "Examples" => "examples.md",
        "Details" => "details.md",
        "Options" => "options.md",
        "NLopt" => "nlopt.md",
        "CUDA" => "cuda.md",
        "Benchmark" => "bench.md"
        "Validation" => "validation.md",
        "API" => "api.md",
        "Citation & Reference" => "ref.md",
    ],
)

deploydocs(repo = "github.com/PharmCat/Metida.jl.git")
