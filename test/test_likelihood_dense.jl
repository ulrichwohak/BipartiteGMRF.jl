@testset "dense likelihood reference" begin
    d = synthetic_data()
    ss = suffstats(BipartiteNormalizedModel, d.f, d.w, d.y;
        weighting=Weighting(observations=:raw),
        standardize=false,
    )
    model = BipartiteNormalizedModel(ss.A_prior; rho_limit=0.99)

    rho = 0.25
    sigma_a = 0.8
    sigma_z = 0.5
    sigma_epsilon = 0.4
    params = [
        atanh(rho / 0.99),
        log(sigma_a),
        log(sigma_z),
        log(sigma_epsilon),
    ]

    obs_stats = BipartiteGMRF.objective_stats(model, ss, params)
    exact_nll = BipartiteGMRF.nll_exact_value(model, ss, params, obs_stats)

    k = ss.K
    n = ss.N_firms + ss.N_workers
    obs_rows = repeat(1:k, 2)
    entity_cols = vcat(ss.base.f, ss.N_firms .+ ss.base.w)
    V = sparse(obs_rows, entity_cols, ones(Float64, 2k), k, n)

    Q = BipartiteGMRF.model_precision(model, rho, sigma_a, sigma_z)
    prior_cov = inv(Matrix(Q))
    Sigma_y = Matrix(V * prior_cov * transpose(V)) + sigma_epsilon^2 * Matrix{Float64}(I, k, k)
    dense_nll = 0.5 * (
        logdet(Symmetric(Sigma_y)) +
        dot(ss.base.y, Sigma_y \ ss.base.y)
    )

    @test exact_nll ≈ dense_nll atol=1e-8 rtol=1e-8
end


@testset "variance-stable paired HutchSLQ logdet" begin
    d_vs = synthetic_data()
    ss = suffstats(BipartiteVarianceStableModel, d_vs.f, d_vs.w, d_vs.y;
        weighting=Weighting(observations=:raw),
        standardize=false,
    )
    model = @test_warn "contains a cycle" BipartiteVarianceStableModel(ss.A_prior)
    solver = HutchSLQ(
        logdet_probes=4096,
        lanczos_iters=20,
        cg_tol=1e-12,
        cg_maxiter=1000,
        optim_iters=2,
    )

    for sigma_z in (0.4, 1e-6)
        params = [
            atanh(0.25 / 0.99),
            log(0.8),
            log(sigma_z),
            log(0.4),
        ]
        obs_stats = BipartiteGMRF.objective_stats(model, ss, params)
        exact_nll = BipartiteGMRF.nll_exact_value(model, ss, params, obs_stats)
        cache = BipartiteGMRF.make_hutch_cache(model, ss, solver)
        hutch_nll = BipartiteGMRF.nll_hutch_value(
            model,
            ss,
            solver,
            params,
            obs_stats,
            cache;
            seed=17,
        )

        @test isfinite(hutch_nll)
        @test hutch_nll ≈ exact_nll atol=0.03
    end
end
