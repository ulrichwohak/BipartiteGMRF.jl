@testset "solver agreement" begin
    df = repeated_df()
    weighting = Weighting(observations=:edge)
    fixed_rho = 0.25

    exact = gmrf_mle(
        df;
        solver=ExactCholesky(optim_iters=80, polish=true),
        weighting=weighting,
        fix_rho=fixed_rho,
        decompose=false,
        seed=11,
    )
    hutch = gmrf_mle(
        df;
        solver=HutchSLQ(
            logdet_probes=300,
            lanczos_iters=8,
            cg_tol=1e-9,
            cg_maxiter=200,
            optim_iters=160,
        ),
        weighting=weighting,
        fix_rho=fixed_rho,
        decompose=false,
        seed=11,
    )

    @test exact.rho == fixed_rho
    @test hutch.rho == fixed_rho
    @test abs(exact.nll - hutch.nll) < 0.02
    @test abs(exact.sigma_z - hutch.sigma_z) < 0.02
    @test abs(exact.sigma_epsilon - hutch.sigma_epsilon) < 0.02
    @test exact.sigma_a < 1e-3
    @test hutch.sigma_a < 1e-3
end
