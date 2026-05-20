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

    p = @test_warn "variance-stable prior no longer guarantees" GMRFProblem(
        synthetic_df();
        prior=VarianceStablePrior(),
    )
    @test_throws ArgumentError solve(p, ExactCholesky(optim_iters=2); decompose=false)

    ps = GMRFProblem(synthetic_df(); prior=SpectralPrior())
    @test_throws ArgumentError solve(ps, ExactCholesky(optim_iters=2); decompose=false)
end
