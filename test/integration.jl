@testset "integration" begin
    result = gmrf_mle(
        synthetic_df();
        prior=NormalizedPrior(),
        solver=ExactCholesky(optim_iters=5, polish=false),
        weighting=Weighting(observations=:raw),
        decompose=3,
        seed=3,
    )
    @test result.model_decomposition !== nothing
    @test StatsAPI.coef(result) == coef(result)
    @test StatsAPI.loglikelihood(result) == -nll(result)
    @test StatsAPI.nobs(result) == result.problem.K

    fd = decompose(result; kind=:fitted, probes=3, seed=3)
    @test isfinite(fd.V_total)

    op = covariance(result; kind=:fitted)
    block = cov_block(op; firms=[1], workers=[10])
    @test size(block.matrix) == (2, 2)
    @test all(isfinite, block.matrix)
end
