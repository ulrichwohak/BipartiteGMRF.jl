using JET

function setup_jet_fixture()
    problem = GMRFProblem(synthetic_df())
    problem_values = Tuple(BipartiteGMRF.problem_fields(problem))
    result = solve(
        problem,
        ExactCholesky(optim_iters=2, polish=false);
        decompose=nothing,
    )
    return problem, problem_values, result
end

function jet_problem_flow(problem_values)
    return GMRFProblem(problem_values...)
end

function jet_exact_solve_flow(problem)
    Q = BipartiteGMRF.model_precision(problem.model, 0.3, 0.8, 0.6)
    M = Q + (1.0 / 0.25^2) .* problem.VtV
    return cholesky(Symmetric(M)) \ problem.projected_y
end

function jet_covariance_flow(result)
    op = covariance(result; kind=:model)
    return cov_block(op; firms=(1,), workers=(10,))
end

@testset "JET" begin
    problem, problem_values, result = setup_jet_fixture()

    JET.@test_opt target_modules=(BipartiteGMRF,) jet_problem_flow(problem_values)
    JET.@test_opt target_modules=(BipartiteGMRF,) jet_exact_solve_flow(problem)
    JET.@test_opt target_modules=(BipartiteGMRF,) jet_covariance_flow(result)
end
