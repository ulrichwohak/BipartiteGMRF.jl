function solve(
    problem::GMRFProblem,
    solver::HutchSLQ;
    decompose::Union{Bool,Nothing,Int}=nothing,
    fix_rho::Union{Nothing,Float64}=nothing,
    seed::Int=42,
    verbose::Bool=false,
)
    fit = optimize_problem(problem, solver; fix_rho=fix_rho, seed=seed, verbose=verbose)
    result = GMRFResult(
        fit.rho,
        fit.sigma_a,
        fit.sigma_z,
        fit.sigma_epsilon,
        fit.rho_eps,
        fit.nll,
        fit.converged,
        fit.iterations,
        fit.obj_evals,
        fit.optimization_time,
        nothing,
        nothing,
        fit.problem,
        fit.problem.prior,
        solver,
        fit.theta_unconstrained,
        fit_result_metadata(fit.problem, fit.rho, fix_rho),
    )
    probes = decompose === true ? 200 : decompose isa Int ? decompose : 0
    if probes > 0
        pd = _decompose_model(result; probes=probes, seed=seed, verbose=verbose)
        result = GMRFResult(
            result.rho, result.sigma_a, result.sigma_z, result.sigma_epsilon,
            result.rho_eps, result.nll, result.converged, result.iterations,
            result.obj_evals, result.optimization_time, pd, result.fitted_decomposition,
            result.problem, result.prior, result.solver, result.theta_unconstrained,
            result.metadata,
        )
    end
    return result
end
