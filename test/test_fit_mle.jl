@testset "fit_mle interface" begin
    d = synthetic_data()

    @testset "fit_mle NormalizedModel" begin
        result = fit_mle(BipartiteNormalizedModel, d.f, d.w, d.y;
            solver=ExactCholesky(optim_iters=5, polish=false),
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
        result = fit_mle(BipartiteUnnormalizedModel, d.f, d.w, d.y;
            solver=ExactCholesky(optim_iters=5, polish=false),
            seed=1,
        )
        @test isfinite(result.nll)
        @test isfinite(result.rho)
        @test result.sigma_a > 0
    end

    @testset "fit_mle VarianceStableModel" begin
        td = tree_data()
        result = fit_mle(BipartiteVarianceStableModel, td.f, td.w, td.y;
            solver=ExactCholesky(optim_iters=5, polish=false),
            seed=1,
            rho_limit=0.95,
        )
        @test isfinite(result.nll)
        @test isfinite(result.rho)
        @test result.sigma_a > 0
    end

    @testset "two-step suffstats + fit_mle" begin
        ss = suffstats(BipartiteNormalizedModel, d.f, d.w, d.y)
        result = fit_mle(BipartiteNormalizedModel, ss;
            solver=ExactCholesky(optim_iters=5, polish=false),
            seed=1,
        )
        @test result.converged || result.nll < Inf
        @test result.sigma_a > 0
        @test result.sigma_z > 0
        @test result.sigma_epsilon > 0
    end

    @testset "suffstats preserves dimensions" begin
        ss = suffstats(BipartiteNormalizedModel, d.f, d.w, d.y)
        @test ss.N_firms == 3
        @test ss.N_workers == 4
        @test ss.K == 8
        @test ss.personyear_rows == 8
        @test size(ss.design.VtV) == (7, 7)
        @test size(ss.A_prior) == (3, 4)
    end

    @testset "rho_limit symbol rejected for non-VS models" begin
        ss = suffstats(BipartiteNormalizedModel, d.f, d.w, d.y)
        @test_throws ArgumentError fit_mle(BipartiteNormalizedModel, ss; rho_limit=:auto)
    end
end
