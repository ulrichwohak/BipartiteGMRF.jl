@testset "fit_mle interface" begin
    df = synthetic_df()

    @testset "fit_mle matches gmrf_mle for NormalizedPrior" begin
        old = gmrf_mle(df;
            prior=NormalizedPrior(),
            solver=ExactCholesky(optim_iters=5, polish=false),
            decompose=false,
            seed=1,
        )
        new = fit_mle(BipartiteNormalizedModel, df;
            solver=ExactCholesky(optim_iters=5, polish=false),
            decompose=false,
            seed=1,
            rho_limit=0.99,
            model_adjacency=:binary,
        )
        @test old.nll ≈ new.nll
        @test old.rho ≈ new.rho
        @test old.sigma_a ≈ new.sigma_a
        @test old.sigma_z ≈ new.sigma_z
        @test old.sigma_epsilon ≈ new.sigma_epsilon
    end

    @testset "fit_mle matches gmrf_mle for UnnormalizedPrior" begin
        old = gmrf_mle(df;
            prior=UnnormalizedPrior(),
            solver=ExactCholesky(optim_iters=5, polish=false),
            decompose=false,
            seed=1,
        )
        new = fit_mle(BipartiteUnnormalizedModel, df;
            solver=ExactCholesky(optim_iters=5, polish=false),
            decompose=false,
            seed=1,
        )
        @test old.nll ≈ new.nll
        @test old.rho ≈ new.rho
        @test old.sigma_a ≈ new.sigma_a
    end

    @testset "fit_mle matches gmrf_mle for VarianceStablePrior" begin
        tdf = tree_df()
        old = gmrf_mle(tdf;
            prior=VarianceStablePrior(rho_limit=0.95),
            solver=ExactCholesky(optim_iters=5, polish=false),
            decompose=false,
            seed=1,
        )
        new = fit_mle(BipartiteVarianceStableModel, tdf;
            solver=ExactCholesky(optim_iters=5, polish=false),
            decompose=false,
            seed=1,
            rho_limit=0.95,
        )
        @test old.nll ≈ new.nll
        @test old.rho ≈ new.rho
        @test old.sigma_a ≈ new.sigma_a
    end

    @testset "two-step suffstats + fit_mle" begin
        ss = suffstats(BipartiteNormalizedModel, df)
        result = fit_mle(BipartiteNormalizedModel, ss;
            solver=ExactCholesky(optim_iters=5, polish=false),
            decompose=false,
            seed=1,
        )
        @test result.converged || result.nll < Inf
        @test result.sigma_a > 0
        @test result.sigma_z > 0
        @test result.sigma_epsilon > 0
    end

    @testset "suffstats preserves dimensions" begin
        ss = suffstats(BipartiteNormalizedModel, df)
        @test ss.N_firms == 3
        @test ss.N_workers == 4
        @test ss.K == 8
        @test ss.personyear_rows == 8
        @test size(ss.VtV) == (7, 7)
        @test size(ss.A_prior) == (3, 4)
    end

    @testset "fit_mle with decompose" begin
        result = fit_mle(BipartiteNormalizedModel, df;
            solver=ExactCholesky(optim_iters=5, polish=false),
            decompose=50,
            seed=1,
        )
        @test result.model_decomposition !== nothing
        @test result.model_decomposition.V_firm > 0
    end
end
