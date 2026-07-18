@testset "solvers" begin
    exact = fitted_exact()
    @test isfinite(exact.nll)
    @test isfinite(exact.rho)
    @test exact.model isa AbstractBipartiteModel
    @test exact.stats isa BipartiteGMRFStats
    @test startswith(sprint(show, exact), "GMRFResult(rho=")
    text_plain = sprint(show, MIME"text/plain"(), exact)
    @test occursin("GMRFResult\n", text_plain)
    @test occursin("parameters:", text_plain)
    @test occursin("rho:", text_plain)
    @test occursin("model:", text_plain)

    hutch = fit_mle(
        BipartiteNormalizedModel,
        synthetic_df();
        solver=HutchSLQ(logdet_probes=2, lanczos_iters=3, optim_iters=2, cg_maxiter=50),
        decompose=false,
        seed=1,
        verbose=false,
    )
    @test isfinite(hutch.nll)

    edge = fit_mle(
        BipartiteNormalizedModel,
        repeated_df();
        solver=ExactCholesky(optim_iters=3, polish=false),
        weighting=Weighting(observations=:edge),
        decompose=false,
        seed=1,
    )
    @test isfinite(edge.nll)

    effective = fit_mle(
        BipartiteNormalizedModel,
        repeated_df();
        solver=ExactCholesky(optim_iters=3, polish=false),
        weighting=Weighting(observations=:effective, rho_eps=:estimate),
        decompose=false,
        seed=1,
    )
    @test isfinite(effective.nll)
    @test 0 <= effective.rho_eps < 1

    ss_limited = suffstats(BipartiteNormalizedModel, synthetic_df())
    model_limited = BipartiteNormalizedModel(ss_limited.A_prior; rho_limit=0.4)
    @test_throws ArgumentError solve(model_limited, ss_limited, ExactCholesky(optim_iters=2, polish=false);
        fix_rho=0.41, decompose=false)
    fixed = solve(model_limited, ss_limited, ExactCholesky(optim_iters=2, polish=false);
        fix_rho=0.2, decompose=false)
    @test fixed.rho ≈ 0.2
    @test abs(fixed.rho) < BipartiteGMRF.rho_limit(model_limited)

    ss_vs = suffstats(BipartiteVarianceStableModel, synthetic_df())
    model_vs = @test_warn "variance-stable model no longer guarantees" BipartiteVarianceStableModel(
        ss_vs.A_prior,
    )
    # VS + ExactCholesky on a cyclic graph is now permitted (was: capability throw);
    # PD is enforced by the caller via rho_limit < 1/lambda_NB.
    rvs = solve(model_vs, ss_vs, ExactCholesky(optim_iters=2); decompose=false, seed=1)
    @test isfinite(rvs.nll)

    ss_sp = suffstats(BipartiteSpectralModel, synthetic_df())
    model_sp = BipartiteSpectralModel(ss_sp.A_prior)
    @test_throws ArgumentError solve(model_sp, ss_sp, ExactCholesky(optim_iters=2); decompose=false)
end

@testset "g_reltol convergence tolerance" begin
    @test HutchSLQ().g_reltol == 1e-7
    @test HutchSLQ(g_reltol=1e-4).g_reltol == 1e-4
    @test_throws ArgumentError HutchSLQ(g_reltol=0.0)
    @test_throws ArgumentError HutchSLQ(g_reltol=-1.0)
    # ExactCholesky also carries g_reltol; its effective tolerance is
    # max(1e-3, g_reltol*|nll0|), so small-problem behaviour is unchanged.
    @test ExactCholesky().g_reltol == 1e-7
    @test ExactCholesky(g_reltol=1e-5).g_reltol == 1e-5
    @test_throws ArgumentError ExactCholesky(g_reltol=0.0)
    @test_throws ArgumentError ExactCholesky(g_reltol=-1.0)

    # The relative tolerance scales the simplex-spread stopping rule by the
    # objective magnitude, so a looser tolerance must converge no later than a
    # tighter one on the same seeded problem; both converge within budget.
    base = (logdet_probes=8, lanczos_iters=10, optim_iters=500, cg_maxiter=200)
    tight = fit_mle(BipartiteNormalizedModel, synthetic_df();
                    solver=HutchSLQ(; base..., g_reltol=1e-7),
                    decompose=false, seed=1, verbose=false)
    loose = fit_mle(BipartiteNormalizedModel, synthetic_df();
                    solver=HutchSLQ(; base..., g_reltol=1e-2),
                    decompose=false, seed=1, verbose=false)
    @test tight.converged
    @test loose.converged
    @test loose.iterations <= tight.iterations
end

@testset "VS + ExactCholesky on acyclic graphs (issue #82)" begin
    # Forest (caterpillar path): ExactCholesky is now allowed for the VS model;
    # exact log-dets sidestep the HutchSLQ small-sigma_z cancellation.
    tdf = tree_df()
    ss_tree = suffstats(BipartiteVarianceStableModel, tdf)
    model_tree = BipartiteVarianceStableModel(ss_tree.A_prior)
    @test BipartiteGMRF.is_forest(model_tree.graph.A)
    res = solve(model_tree, ss_tree, ExactCholesky(optim_iters=200, polish=true); decompose=false, seed=1)
    @test res.converged
    @test isfinite(res.nll)
    @test isfinite(res.rho)
    @test res.sigma_a > 0 && res.sigma_z > 0 && res.sigma_epsilon > 0

    # Cyclic graph: construction warns by default (strict_forest=false).
    ss_cyc = suffstats(BipartiteVarianceStableModel, synthetic_df())
    model_cyc = @test_warn "contains a cycle" BipartiteVarianceStableModel(ss_cyc.A_prior)
    @test !BipartiteGMRF.is_forest(model_cyc.graph.A)
    # strict_forest=true still errors at construction on a cyclic graph.
    @test_throws ArgumentError BipartiteVarianceStableModel(ss_cyc.A_prior; strict_forest=true)
    # ExactCholesky is now permitted on cyclic graphs (PD enforced via rho_limit <
    # 1/lambda_NB); the old capability throw is gone and the solver returns a result.
    rce = solve(model_cyc, ss_cyc, ExactCholesky(optim_iters=10); decompose=false, seed=1)
    @test isfinite(rce.nll)
    # HutchSLQ remains available on cyclic graphs.
    resh = solve(model_cyc, ss_cyc, HutchSLQ(logdet_probes=4, lanczos_iters=6, optim_iters=10, cg_maxiter=50);
                 decompose=false, seed=1)
    @test isfinite(resh.nll)
end
