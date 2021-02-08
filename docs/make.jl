using Documenter, Metida

makedocs(
    modules = [Metida],
    sitename = "Metida Documentation",
    authors = "Vladimir Arnautov",
    linkcheck = true,
    doctest = true,
    pages = [
        "Home" => "index.md",
        "User guide" => [
        "First step" => "instanduse.md",
        "Examples" => "examples.md",
        "NLopt" => "nlopt.md",
        "CUDA" => "cuda.md",
        ],
        "Details" => "details.md",
        "Benchmark" => "bench.md",
        "Validation" => "validation.md",
        "API" => "api.md",
        "Citation & Reference" => "ref.md",
    ],
)

deploydocs(repo = "github.com/PharmCat/Metida.jl.git",
    devurl = "dev",
    versions = ["stable" => "v^", "v#.#.#", "dev" => "dev"],
)
