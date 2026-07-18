function solve(
    model::AbstractBipartiteModel,
    stats::BipartiteGMRFStats,
    solver::HutchSLQ;
    decompose::Union{Nothing,Int}=nothing,
    fix_rho::Union{Nothing,Float64}=nothing,
    seed::Int=42,
    verbose::Bool=false,
)
    fit = optimize_problem(model, stats, solver; fix_rho=fix_rho, seed=seed, verbose=verbose)
    result = build_gmrf_result(fit, solver, fix_rho)
    return attach_model_decomposition(result, decompose, seed, verbose)
end
