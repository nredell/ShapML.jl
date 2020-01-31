using Documenter, ShapML

makedocs(
    sitename = "ShapML",
    authors = "Nickalus Redell",
    doctest = false,
    format = Documenter.HTML(
        highlights = ["yaml"]
        ),
    pages = [
        "Introduction" => "index.md",
        "Vignettes" => Any["Stochastic vs. TreeSHAP" => "vignettes/consistency.md"],
        "Functions" => "functions/functions.md"
    ]
)

deploydocs(
    repo = "github.com/nredell/ShapML.jl.git"
)
