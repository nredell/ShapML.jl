using Documenter, ShapML

makedocs(
sitename = "ShapML",
authors = "Nickalus Redell",
doctest = false,
pages = [
    "Introduction" => "index.md",
    ]
)

deploydocs(
    repo = "github.com/nredell/ShapML.jl.git",
)
