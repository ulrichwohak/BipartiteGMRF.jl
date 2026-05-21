@testset "solver agreement" begin
    @testset "fixed rho" begin
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

    @testset "free rho" begin
        # This is a deterministic single-seed solver agreement check, not a
        # Monte Carlo recovery criterion. Seed 52 exercises an interior
        # free-rho optimum with nonzero sigma_a on a panel small enough for CI.
        # The distinct truth tuple keeps this separate from recovery debt.
        truth = (rho=0.35, sigma_a=0.8, sigma_z=0.6, sigma_epsilon=0.25)
        df, _ = simulate_gmrf_panel(
            52;
            n_firms=150,
            n_workers=150,
            n_edges=3500,
            reps=4,
            truth=truth,
        )
        problem = GMRFProblem(df; standardize=false)
        exact = solve(
            problem,
            ExactCholesky(optim_iters=160, polish=true);
            decompose=false,
            seed=52,
        )
        hutch = solve(
            problem,
            HutchSLQ(
                # Finite probes trade stochastic precision for CI runtime.
                logdet_probes=400,
                lanczos_iters=16,
                cg_tol=1e-9,
                cg_maxiter=300,
                optim_iters=220,
            );
            decompose=false,
            seed=52,
        )

        @test exact.converged
        @test hutch.converged
        @test exact.sigma_a > 1e-3
        @test hutch.sigma_a > 1e-3
        # Julia 1.10 resolves an older solver/test stack with a larger
        # finite-probe rho/nll drift for this deterministic HutchSLQ path.
        @test abs(exact.rho - hutch.rho) < 0.20
        @test abs(exact.sigma_a - hutch.sigma_a) < 0.03
        @test abs(exact.sigma_z - hutch.sigma_z) < 0.03
        @test abs(exact.sigma_epsilon - hutch.sigma_epsilon) < 0.03
        @test abs(exact.nll - hutch.nll) < 0.50
    end
end
