@testset "fit_mle interface" begin
    df = synthetic_df()

    @testset "fit_mle NormalizedModel" begin
        result = fit_mle(BipartiteNormalizedModel, df;
            solver=ExactCholesky(optim_iters=5, polish=false),
            decompose=false,
            seed=1,
            rho_limit=0.99,
            model_adjacency=:binary,
        )
        @test isfinite(result.nll)
        @test isfinite(result.rho)
        @test result.sigma_a > 0
        @test result.sigma_z > 0
        @test result.sigma_epsilon > 0
    end

    @testset "fit_mle UnnormalizedModel" begin
        result = fit_mle(BipartiteUnnormalizedModel, df;
            solver=ExactCholesky(optim_iters=5, polish=false),
            decompose=false,
            seed=1,
        )
        @test isfinite(result.nll)
        @test isfinite(result.rho)
        @test result.sigma_a > 0
    end

    @testset "fit_mle VarianceStableModel" begin
        tdf = tree_df()
        result = fit_mle(BipartiteVarianceStableModel, tdf;
            solver=ExactCholesky(optim_iters=5, polish=false),
            decompose=false,
            seed=1,
            rho_limit=0.95,
        )
        @test isfinite(result.nll)
        @test isfinite(result.rho)
        @test result.sigma_a > 0
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
