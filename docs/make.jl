using BipartiteGMRF
using Documenter

makedocs(;
    modules = [BipartiteGMRF],
    sitename = "BipartiteGMRF.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://ulrichwohak.github.io/BipartiteGMRF.jl",
        edit_link = "main",
        repolink = "https://github.com/ulrichwohak/BipartiteGMRF.jl",
    ),
    pages = [
        "Overview" => "index.md",
        "API" => "api.md",
        "Models" => "models.md",
        "Solvers" => "solvers.md",
        "Weighting" => "weighting.md",
        "Covariance" => "covariance.md",
        "Non-Backtracking" => "nonbacktracking.md",
        "Examples" => "examples.md",
        "Performance" => "performance.md",
    ],
    checkdocs = :exports,
    warnonly = [:missing_docs, :cross_references, :docs_block],
)

deploydocs(;
    repo = "github.com/ulrichwohak/BipartiteGMRF.jl.git",
    devbranch = "main",
)
