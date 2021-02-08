using Documenter, Metida

makedocs(
    modules = [Metida],
    sitename = "Metida Documentation",
    authors = "Vladimir Arnautov",
    linkcheck = false,
    doctest = false,
    pages = [
        "Home" => "index.md",
        "Details" => "details.md",
        "Examples" => "examples.md",
        "Options" => "options.md",
        "NLopt" => "nlopt.md",
        "CUDA" => "cuda.md",
        "Benchmark" => "bench.md",
        "Validation" => "validation.md",
        "API" => "api.md",
        "Citation & Reference" => "ref.md",
    ],
)

deploydocs(repo = "github.com/PharmCat/Metida.jl.git",
    versions = ["stable" => "v^", "v#.#.#", devurl => devurl]
)
