using Documenter, Metida

makedocs(
    modules = [Metida],
    sitename = "Metida.jl",
    authors = "Vladimir Arnautov",
    pages = [
        "Home" => "index.md",
        "User guide" => [
        "First step" => "instanduse.md",
        "Examples" => "examples.md",
        "Bioequivalence" => "bioequivalence.md",
        "NLopt" => "nlopt.md",
        "CUDA" => "cuda.md",
        "F.A.Q." => "faq.md",
        ],
        "Details" => "details.md",
        "Benchmark" => "bench.md",
        "Validation" => "validation.md",
        "API" => "api.md",
        "Citation & Reference" => "ref.md",
    ],
)

deploydocs(repo = "github.com/PharmCat/Metida.jl.git", push_preview = true,
)
