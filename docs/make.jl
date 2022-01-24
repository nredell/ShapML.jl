using ShapML
using Documenter

DocMeta.setdocmeta!(
    ShapML, :DocTestSetup, :(using ShapML); recursive=true
)

makedocs(
    sitename = "ShapML",
    authors = "Nickalus Redell",
    doctest = true,
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
    repo = "github.com/nredell/ShapML.jl",
    devbranch="master",
)
