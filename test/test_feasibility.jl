@testset "variance-stable feasibility" begin
    m_default = BipartiteVarianceStableModel(sparse([1.0], [1.0], [1.0], 1, 1))
    @test BipartiteGMRF.rho_limit(m_default) == 0.99
    m_auto = BipartiteVarianceStableModel(sparse([1.0], [1.0], [1.0], 1, 1); rho_limit=:auto)
    @test m_auto.rho_limit == :auto
    @test_throws ArgumentError BipartiteVarianceStableModel(sparse([1.0], [1.0], [1.0], 1, 1); rho_limit=:invalid)
    @test_throws ArgumentError BipartiteVarianceStableModel(sparse([1.0], [1.0], [1.0], 1, 1); rho_limit=1.0)

    auto_forest_result = @test_logs (
        :info,
        r"VS feasibility resolved: lambda_NB=0.0000, rho_ceiling=1.0000, rho_limit=0.9900, source=auto",
    ) match_mode=:any fit_mle(
        BipartiteVarianceStableModel,
        tree_df();
        rho_limit=:auto,
        solver=ExactCholesky(optim_iters=2, polish=false),
        decompose=false,
        seed=7,
    )
    @test BipartiteGMRF.rho_limit(auto_forest_result.model) == 0.99
    @test auto_forest_result.stats.metadata.rho_limit_source == :auto
    @test auto_forest_result.stats.metadata.rho_ceiling == 1.0
    @test auto_forest_result.stats.metadata.resolved_rho_limit == 0.99
    @test auto_forest_result.stats.metadata.nb_spectrum isa NBSpectrum
    @test feasibility(auto_forest_result.model, auto_forest_result.stats).safe

    edges = [(firm, worker) for firm in 1:3 for worker in 1:3]
    cyclic_df = DataFrame(
        firm_id=first.(edges),
        worker_id=last.(edges),
        y=collect(1.0:length(edges)),
    )
    auto_cyclic_result = @test_logs (
        :info,
        r"VS feasibility resolved: lambda_NB=2.0000, rho_ceiling=0.5000, rho_limit=0.4900, source=auto",
    ) (
        :warn,
        r"contains a cycle",
    ) match_mode=:any fit_mle(
        BipartiteVarianceStableModel,
        cyclic_df;
        rho_limit=:auto,
        solver=ExactCholesky(optim_iters=2, polish=false),
        standardize=false,
        decompose=false,
        seed=7,
    )
    @test BipartiteGMRF.rho_limit(auto_cyclic_result.model) ≈ 0.49
    auto_report = feasibility(auto_cyclic_result.model, auto_cyclic_result.stats)
    @test auto_report.lambda_nb ≈ 2.0 atol=1e-10
    @test auto_report.rho_ceiling ≈ 0.5 atol=1e-10
    @test auto_report.source == :auto
    @test auto_report.safe

    explicit_ss = suffstats(BipartiteVarianceStableModel, cyclic_df; standardize=false)
    explicit_model = @test_warn "contains a cycle" BipartiteVarianceStableModel(
        explicit_ss.A_prior; rho_limit=0.9,
    )
    original_limit = BipartiteGMRF.rho_limit(explicit_model)
    params = [atanh(0.2 / original_limit), log(0.8), log(0.6), log(0.5)]
    obs_stats = BipartiteGMRF.objective_stats(explicit_model, explicit_ss, params)
    nll_before_audit = BipartiteGMRF.nll_exact_value(explicit_model, explicit_ss, params, obs_stats)
    explicit_report = @test_warn "explicit rho_limit=0.9000" feasibility(explicit_model, explicit_ss)
    @test !explicit_report.safe
    @test explicit_report.source == :explicit
    @test explicit_report.recommended_rho_limit ≈ 0.49
    @test BipartiteGMRF.rho_limit(explicit_model) == original_limit
    @test get(explicit_ss.metadata, :nb_spectrum, nothing) === nothing
    @test BipartiteGMRF.nll_exact_value(explicit_model, explicit_ss, params, obs_stats) == nll_before_audit

    norm_ss = suffstats(BipartiteNormalizedModel, synthetic_df())
    norm_model = BipartiteNormalizedModel(norm_ss.A_prior)
    @test_throws ArgumentError feasibility(norm_model, norm_ss)

    fixed_result = @test_logs (
        :info,
        r"VS fit status: .*status=fixed",
    ) solve(
        auto_forest_result.model,
        auto_forest_result.stats,
        ExactCholesky(optim_iters=2, polish=false);
        fix_rho=0.2,
        decompose=false,
        seed=7,
    )
    @test !rho_at_bound(fixed_result)
    @test fixed_result.metadata.rho_status == :fixed
    @test occursin("rho_status=fixed", sprint(show, fixed_result))
    @test occursin("rho status: fixed", sprint(show, MIME"text/plain"(), fixed_result))

    bound_metadata = @test_logs (
        :warn,
        r"VS fit status: .*status=bound_censored",
    ) BipartiteGMRF.fit_result_metadata(auto_cyclic_result.model, 0.99 * BipartiteGMRF.rho_limit(auto_cyclic_result.model), nothing)
    bound_result = GMRFResult(
        0.99 * BipartiteGMRF.rho_limit(auto_cyclic_result.model),
        fixed_result.sigma_a,
        fixed_result.sigma_z,
        fixed_result.sigma_epsilon,
        fixed_result.rho_eps,
        fixed_result.nll,
        fixed_result.converged,
        fixed_result.iterations,
        fixed_result.obj_evals,
        fixed_result.optimization_time,
        nothing,
        nothing,
        auto_cyclic_result.model,
        auto_cyclic_result.stats,
        fixed_result.solver,
        fixed_result.theta_unconstrained,
        bound_metadata,
    )
    @test rho_at_bound(bound_result)
    @test occursin("rho_status=bound_censored", sprint(show, bound_result))
end
