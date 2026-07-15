push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

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
        "Priors" => "priors.md",
        "Solvers" => "solvers.md",
        "Weighting" => "weighting.md",
        "Covariance" => "covariance.md",
        "Non-Backtracking" => "nonbacktracking.md",
        "NB-Walk Theory" => "nb_walk_theory.md",
        "Examples" => "examples.md",
        "Performance" => "performance.md",
    ],
    checkdocs = :exports,
)

deploydocs(;
    repo = "github.com/ulrichwohak/BipartiteGMRF.jl.git",
    devbranch = "main",
)
