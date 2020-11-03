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
    ],
)

deploydocs(repo = "github.com/PharmCat/Metida.jl.git")
