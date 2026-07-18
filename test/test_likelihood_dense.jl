@testset "dense likelihood reference" begin
    problem = GMRFProblem(
        synthetic_df();
        model_type=BipartiteNormalizedModel,
        weighting=Weighting(observations=:raw),
        standardize=false,
    )

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

    stats = BipartiteGMRF.objective_stats(problem, params)
    exact_nll = BipartiteGMRF.nll_exact_value(problem, params, stats)

    k = problem.K
    n = problem.N_firms + problem.N_workers
    obs_rows = repeat(1:k, 2)
    entity_cols = vcat(problem.base_f_rows, problem.N_firms .+ problem.base_w_cols)
    V = sparse(obs_rows, entity_cols, ones(Float64, 2k), k, n)

    Q = BipartiteGMRF.model_precision(problem.model, rho, sigma_a, sigma_z)
    prior_cov = inv(Matrix(Q))
    Sigma_y = Matrix(V * prior_cov * transpose(V)) + sigma_epsilon^2 * Matrix{Float64}(I, k, k)
    dense_nll = 0.5 * (
        logdet(Symmetric(Sigma_y)) +
        dot(problem.y, Sigma_y \ problem.y)
    )

    @test exact_nll ≈ dense_nll atol=1e-8 rtol=1e-8
end


@testset "variance-stable paired HutchSLQ logdet" begin
    problem = @test_warn "contains a cycle" GMRFProblem(
        synthetic_df();
        model_type=BipartiteVarianceStableModel,
        weighting=Weighting(observations=:raw),
        standardize=false,
    )
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
        stats = BipartiteGMRF.objective_stats(problem, params)
        exact_nll = BipartiteGMRF.nll_exact_value(problem, params, stats)
        cache = BipartiteGMRF.make_hutch_cache(problem, solver)
        hutch_nll = BipartiteGMRF.nll_hutch_value(
            problem,
            solver,
            params,
            stats,
            cache;
            seed=17,
        )

        @test isfinite(hutch_nll)
        @test hutch_nll ≈ exact_nll atol=0.03
    end
end
