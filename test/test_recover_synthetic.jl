@testset "synthetic recovery" begin
    # TODO #58: recovery remains deferred pending amended acceptance criteria.
    seeds = [203]
    estimates = map(seeds) do seed
        df, _ = simulate_gmrf_panel(seed)
        fit_mle(
            BipartiteNormalizedModel,
            df;
            standardize=false,
            solver=ExactCholesky(optim_iters=160, polish=true),
            decompose=false,
            seed=seed,
        )
    end
    _, truth = simulate_gmrf_panel(first(seeds))

    rhos = [estimate.rho for estimate in estimates]
    sigma_as = [estimate.sigma_a for estimate in estimates]
    sigma_zs = [estimate.sigma_z for estimate in estimates]
    sigma_epsilons = [estimate.sigma_epsilon for estimate in estimates]

    @test all(rho -> abs(rho - truth.rho) < 0.10, rhos)
    @test all(s -> abs(log(s) - log(truth.sigma_a)) < 0.15, sigma_as)
    @test all(s -> abs(log(s) - log(truth.sigma_z)) < 0.15, sigma_zs)
    @test all(s -> abs(log(s) - log(truth.sigma_epsilon)) < 0.10, sigma_epsilons)
end
