@testset "solver agreement" begin
    @testset "fixed rho" begin
        rep = repeated_data()
        weighting = Weighting(observations=:edge)
        fixed_rho = 0.25

        exact = fit_mle(
            BipartiteNormalizedModel, rep.f, rep.w, rep.y;
            solver=ExactCholesky(optim_iters=80, polish=true),
            weighting=weighting,
            fix_rho=fixed_rho,
            seed=11,
        )
        hutch = fit_mle(
            BipartiteNormalizedModel, rep.f, rep.w, rep.y;
            solver=HutchSLQ(
                logdet_probes=300,
                lanczos_iters=8,
                cg_tol=1e-9,
                cg_maxiter=200,
                optim_iters=160,
            ),
            weighting=weighting,
            fix_rho=fixed_rho,
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
        data, _ = simulate_gmrf_panel(
            52;
            n_firms=150,
            n_workers=150,
            n_edges=3500,
            reps=4,
            truth=truth,
        )
        ss = suffstats(BipartiteNormalizedModel, data.f, data.w, data.y; standardize=false)
        model = BipartiteNormalizedModel(ss.A_prior; rho_limit=0.99)
        exact = solve(
            model,
            ss,
            ExactCholesky(optim_iters=160, polish=true);
            seed=52,
        )
        hutch = solve(
            model,
            ss,
            HutchSLQ(
                # Finite probes trade stochastic precision for CI runtime.
                logdet_probes=400,
                lanczos_iters=16,
                cg_tol=1e-9,
                cg_maxiter=300,
                optim_iters=220,
            );
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
