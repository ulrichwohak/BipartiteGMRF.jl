@testset "integration" begin
    result = gmrf_mle(
        synthetic_df();
        prior=NormalizedPrior(),
        solver=ExactCholesky(optim_iters=5, polish=false),
        weighting=Weighting(observations=:raw),
        decompose=3,
        seed=3,
    )
    @test result.prior_decomposition !== nothing
    @test StatsAPI.coef(result) == coef(result)
    @test StatsAPI.loglikelihood(result) == -nll(result)
    @test StatsAPI.nobs(result) == result.problem.K

    posterior = posterior_decomposition(result; probes=3, seed=3)
    @test isfinite(posterior.V_total)

    op = posterior_covariance(result)
    block = cov_block(op; firms=[1], workers=[10])
    @test size(block.matrix) == (2, 2)
    @test all(isfinite, block.matrix)
end
