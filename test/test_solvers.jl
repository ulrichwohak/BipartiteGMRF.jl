@testset "solvers" begin
    exact = fitted_exact()
    @test isfinite(exact.nll)
    @test isfinite(exact.rho)
    @test exact.problem isa GMRFProblem
    @test startswith(sprint(show, exact), "GMRFResult(rho=")
    text_plain = sprint(show, MIME"text/plain"(), exact)
    @test occursin("GMRFResult\n", text_plain)
    @test occursin("parameters:", text_plain)
    @test occursin("rho:", text_plain)
    @test occursin("model:", text_plain)

    hutch = gmrf_mle(
        synthetic_df();
        solver=HutchSLQ(logdet_probes=2, lanczos_iters=3, optim_iters=2, cg_maxiter=50),
        decompose=false,
        seed=1,
        verbose=false,
    )
    @test isfinite(hutch.nll)

    edge = gmrf_mle(
        repeated_df();
        solver=ExactCholesky(optim_iters=3, polish=false),
        weighting=Weighting(observations=:edge),
        decompose=false,
        seed=1,
    )
    @test isfinite(edge.nll)

    effective = gmrf_mle(
        repeated_df();
        solver=ExactCholesky(optim_iters=3, polish=false),
        weighting=Weighting(observations=:effective, rho_eps=:estimate),
        decompose=false,
        seed=1,
    )
    @test isfinite(effective.nll)
    @test 0 <= effective.rho_eps < 1

    limited = GMRFProblem(synthetic_df(); prior=NormalizedPrior(rho_limit=0.4))
    @test_throws ArgumentError solve(limited, ExactCholesky(optim_iters=2, polish=false);
        fix_rho=0.41, decompose=false)
    fixed = solve(limited, ExactCholesky(optim_iters=2, polish=false);
        fix_rho=0.2, decompose=false)
    @test fixed.rho ≈ 0.2
    @test abs(fixed.rho) < limited.prior.rho_limit

    p = @test_warn "variance-stable model no longer guarantees" GMRFProblem(
        synthetic_df();
        prior=VarianceStablePrior(),
    )
    # VS + ExactCholesky on a cyclic graph is now permitted (was: capability throw);
    # PD is enforced by the caller via rho_limit < 1/lambda_NB.
    rvs = solve(p, ExactCholesky(optim_iters=2); decompose=false, seed=1)
    @test isfinite(rvs.nll)

    ps = GMRFProblem(synthetic_df(); prior=SpectralPrior())
    @test_throws ArgumentError solve(ps, ExactCholesky(optim_iters=2); decompose=false)
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
    tight = gmrf_mle(synthetic_df(); solver=HutchSLQ(; base..., g_reltol=1e-7),
                     decompose=false, seed=1, verbose=false)
    loose = gmrf_mle(synthetic_df(); solver=HutchSLQ(; base..., g_reltol=1e-2),
                     decompose=false, seed=1, verbose=false)
    @test tight.converged
    @test loose.converged
    @test loose.iterations <= tight.iterations
end

@testset "VS + ExactCholesky on acyclic graphs (issue #82)" begin
    # Forest (caterpillar path): ExactCholesky is now allowed for the VS prior;
    # exact log-dets sidestep the HutchSLQ small-sigma_z cancellation.
    ptree = GMRFProblem(tree_df(); prior=VarianceStablePrior(),
                        weighting=Weighting(observations=:raw))
    @test BipartiteGMRF.is_forest(ptree.A_prior)
    res = solve(ptree, ExactCholesky(optim_iters=200, polish=true); decompose=false, seed=1)
    @test res.converged
    @test isfinite(res.nll)
    @test isfinite(res.rho)
    @test res.sigma_a > 0 && res.sigma_z > 0 && res.sigma_epsilon > 0

    # Cyclic graph: construction warns by default (strict_forest=false).
    pcyc = @test_warn "contains a cycle" GMRFProblem(synthetic_df();
        prior=VarianceStablePrior(), weighting=Weighting(observations=:raw))
    @test !BipartiteGMRF.is_forest(pcyc.A_prior)
    # strict_forest=true still errors at construction on a cyclic graph.
    @test_throws ArgumentError GMRFProblem(synthetic_df();
        prior=VarianceStablePrior(strict_forest=true), weighting=Weighting(observations=:raw))
    # ExactCholesky is now permitted on cyclic graphs (PD enforced via rho_limit <
    # 1/lambda_NB); the old capability throw is gone and the solver returns a result.
    rce = solve(pcyc, ExactCholesky(optim_iters=10); decompose=false, seed=1)
    @test isfinite(rce.nll)
    # HutchSLQ remains available on cyclic graphs.
    resh = solve(pcyc, HutchSLQ(logdet_probes=4, lanczos_iters=6, optim_iters=10, cg_maxiter=50);
                 decompose=false, seed=1)
    @test isfinite(resh.nll)
end
