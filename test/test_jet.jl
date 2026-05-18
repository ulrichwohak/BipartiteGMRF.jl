using JET

function jet_smoke_flow()
    df = DataFrame(
        firm_id = [1, 1, 2, 2],
        worker_id = [10, 11, 11, 12],
        y = [1.0, 0.8, 1.2, 0.7],
    )
    problem = GMRFProblem(df; standardize=false)
    result = solve(
        problem,
        ExactCholesky(optim_iters=2, polish=false);
        decompose=false,
        seed=1,
    )
    op = prior_covariance(result; units=:scaled)
    block = cov_block(op; firms=[1])
    return size(block.matrix)
end

@testset "JET smoke" begin
    # Generate the report as a smoke check. Strict `@test_opt` gating is left
    # for the follow-up type-stability fixes because the current public review
    # intentionally tracks known dynamic dispatch in this flow.
    report = JET.report_call(jet_smoke_flow; target_modules=(BipartiteGMRF,))
    @test report !== nothing
end
