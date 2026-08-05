@testset "integration" begin
    d = synthetic_data()
    result = fit_mle(
        BipartiteNormalizedModel, d.f, d.w, d.y;
        solver=ExactCholesky(optim_iters=5, polish=false),
        weighting=Weighting(observations=:raw),
        seed=3,
    )
    md = decompose(result; kind=:model, probes=3, seed=3)
    @test isfinite(md.V_total)
    @test StatsAPI.coef(result) == coef(result)
    n_dims = result.stats.personyear_rows
    @test StatsAPI.loglikelihood(result) ≈
        -(nll(result) + n_dims * log(result.stats.y_std) + 0.5 * n_dims * log(2pi))
    @test StatsAPI.nobs(result) == result.stats.K

    fd = decompose(result; kind=:fitted, probes=3, seed=3)
    @test isfinite(fd.V_total)

    op = covariance(result; kind=:fitted)
    block = cov_block(op; firms=[1], workers=[1])
    @test size(block.matrix) == (2, 2)
    @test all(isfinite, block.matrix)
end
