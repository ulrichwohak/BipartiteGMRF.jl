@testset "variance-stable feasibility" begin
    @test VarianceStablePrior().rho_limit == 0.99
    @test VarianceStablePrior(rho_limit=:auto).rho_limit == :auto
    @test_throws ArgumentError VarianceStablePrior(rho_limit=:invalid)
    @test_throws ArgumentError VarianceStablePrior(false, 1.0)
    @test_throws ArgumentError VarianceStablePrior(false, :invalid)

    auto_forest = @test_logs (
        :info,
        r"VS feasibility resolved: lambda_NB=0.0000, rho_ceiling=1.0000, rho_limit=0.9900, source=auto",
    ) GMRFProblem(
        tree_df();
        prior=VarianceStablePrior(rho_limit=:auto),
        standardize=false,
    )
    @test auto_forest.prior.rho_limit == 0.99
    @test auto_forest.metadata.rho_limit_source == :auto
    @test auto_forest.metadata.rho_ceiling == 1.0
    @test auto_forest.metadata.resolved_rho_limit == 0.99
    @test auto_forest.metadata.nb_spectrum isa NBSpectrum
    @test feasibility(auto_forest).safe

    edges = [(firm, worker) for firm in 1:3 for worker in 1:3]
    cyclic_df = DataFrame(
        firm_id=first.(edges),
        worker_id=last.(edges),
        y=collect(1.0:length(edges)),
    )
    auto_cyclic = @test_logs (
        :info,
        r"VS feasibility resolved: lambda_NB=2.0000, rho_ceiling=0.5000, rho_limit=0.4900, source=auto",
    ) (
        :warn,
        r"contains a cycle",
    ) GMRFProblem(
        cyclic_df;
        prior=VarianceStablePrior(rho_limit=:auto),
        standardize=false,
    )
    @test auto_cyclic.prior.rho_limit ≈ 0.49
    auto_report = feasibility(auto_cyclic)
    @test auto_report.lambda_nb ≈ 2.0 atol=1e-10
    @test auto_report.rho_ceiling ≈ 0.5 atol=1e-10
    @test auto_report.source == :auto
    @test auto_report.safe

    explicit = @test_warn "contains a cycle" GMRFProblem(
        cyclic_df;
        prior=VarianceStablePrior(rho_limit=0.9),
        standardize=false,
    )
    original_limit = explicit.prior.rho_limit
    params = [atanh(0.2 / original_limit), log(0.8), log(0.6), log(0.5)]
    stats = BipartiteGMRF.objective_stats(explicit, params)
    nll_before_audit = BipartiteGMRF.nll_exact_value(explicit, params, stats)
    explicit_report = @test_warn "explicit rho_limit=0.9000" feasibility(explicit)
    @test !explicit_report.safe
    @test explicit_report.source == :explicit
    @test explicit_report.recommended_rho_limit ≈ 0.49
    @test explicit.prior.rho_limit == original_limit
    @test explicit.metadata.nb_spectrum === nothing
    @test BipartiteGMRF.nll_exact_value(explicit, params, stats) == nll_before_audit

    @test_throws ArgumentError feasibility(GMRFProblem(synthetic_df()))

    fixed_result = @test_logs (
        :info,
        r"VS fit status: .*status=fixed",
    ) solve(
        auto_forest,
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
    ) BipartiteGMRF.fit_result_metadata(auto_cyclic, 0.99 * auto_cyclic.prior.rho_limit, nothing)
    bound_result = GMRFResult(
        0.99 * auto_cyclic.prior.rho_limit,
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
        auto_cyclic,
        auto_cyclic.prior,
        fixed_result.solver,
        fixed_result.theta_unconstrained,
        bound_metadata,
    )
    @test rho_at_bound(bound_result)
    @test occursin("rho_status=bound_censored", sprint(show, bound_result))
end
