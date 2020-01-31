using Documenter, ShapML

makedocs(
sitename = "ShapML",
authors = "Nickalus Redell",
doctest = false,
pages = [
    "Introduction" => "index.md",
    "Algorithm comparison" => "consistency.md"
    ]
)

deploydocs(
    repo = "github.com/nredell/ShapML.jl.git",
)
