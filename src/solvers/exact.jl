"""
    solve(model::AbstractBipartiteModel, stats::BipartiteGMRFStats, solver::AbstractGMRFSolver;
          decompose=nothing, fix_rho=nothing, seed=42, verbose=false)

Fit a bipartite GMRF by maximum likelihood with the given solver. Extends
`CommonSolve.solve`, so it composes with `using LinearSolve` or other
SciML-style packages without name clashes. [`fit_mle`](@ref) is the
higher-level entry point and delegates here.

`decompose` controls the optional model variance decomposition attached to
the result: pass `nothing` (default) to skip it, or a positive integer probe
count to run it. `fix_rho` fixes the local-dependence parameter during
optimization.
"""
function solve(
    model::AbstractBipartiteModel,
    stats::BipartiteGMRFStats,
    solver::ExactCholesky;
    decompose::Union{Nothing,Int}=nothing,
    fix_rho::Union{Nothing,Float64}=nothing,
    seed::Int=42,
    verbose::Bool=false,
)
    fit = optimize_problem(model, stats, solver; fix_rho=fix_rho, seed=seed, verbose=verbose)
    result = build_gmrf_result(fit, solver, fix_rho)
    return attach_model_decomposition(result, decompose, seed, verbose)
end
