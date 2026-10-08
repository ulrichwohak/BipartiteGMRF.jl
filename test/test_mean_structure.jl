@testset "mean structure (profiled Xβ)" begin
    td = tree_data()

    @testset "X=nothing reproduces current estimates" begin
        r0 = fit_mle(BipartiteVarianceStableModel, td.f, td.w, td.y;
            solver=ExactCholesky(optim_iters=5, polish=false), seed=1)
        rX = fit_mle(BipartiteVarianceStableModel, td.f, td.w, td.y;
            X=nothing,
            solver=ExactCholesky(optim_iters=5, polish=false), seed=1)
        @test rX.rho ≈ r0.rho
        @test rX.sigma_a ≈ r0.sigma_a
        @test rX.sigma_z ≈ r0.sigma_z
        @test rX.sigma_epsilon ≈ r0.sigma_epsilon
        @test rX.beta === nothing
    end

    @testset "intercept-only β ≈ 0 when standardized" begin
        K = length(td.y)
        X_int = ones(K, 1)
        r = fit_mle(BipartiteVarianceStableModel, td.f, td.w, td.y;
            X=X_int,
            solver=ExactCholesky(optim_iters=5, polish=false), seed=1)
        @test r.beta !== nothing
        @test length(r.beta) == 1
        # β₀ should be small because standardization already centers y
        @test abs(r.beta[1]) < 0.5
    end

    @testset "known-β recovery" begin
        using Random
        rng = MersenneTwister(77)
        n_firms, n_workers = 20, 20
        A_tree = sparse(
            vcat(1:n_firms, 2:n_firms),
            vcat(1:n_firms, 1:(n_firms-1)),
            ones(2*n_firms - 1), n_firms, n_workers,
        )
        model = BipartiteNormalizedModel(A_tree)
        K = 500
        f_ids = rand(rng, 1:n_firms, K)
        w_ids = rand(rng, 1:n_workers, K)

        truth_beta = [2.0, 0.5, -0.3]
        X = hcat(ones(K), Float64.(f_ids), Float64.(w_ids))

        sim = simulate(model, f_ids, w_ids;
            ρ=0.3, σ_a=0.5, σ_z=0.3, σ_ε=0.5,
            X=X, β=truth_beta, rng=rng)

        r = fit_mle(BipartiteNormalizedModel, f_ids, w_ids, sim.y;
            X=X, n_firms=n_firms, n_workers=n_workers,
            standardize=false,
            solver=ExactCholesky(optim_iters=200, polish=true), seed=1)
        @test r.beta !== nothing
        @test length(r.beta) == 3
        # slope coefficients should be close (intercept exact with standardize=false)
        for j in 1:3
            @test abs(r.beta[j] - truth_beta[j]) < 2.0
        end
    end

    @testset "coef and coefnames include β" begin
        K = length(td.y)
        X_deg = hcat(ones(K), Float64.(td.f))
        r = fit_mle(BipartiteVarianceStableModel, td.f, td.w, td.y;
            X=X_deg,
            solver=ExactCholesky(optim_iters=5, polish=false), seed=1)
        c = coef(r)
        @test length(c) == 4 + 2  # rho, sa, sz, se + 2 betas
        cn = coefnames(r)
        @test cn[end-1] == "beta_1"
        @test cn[end] == "beta_2"
        p = params(r)
        @test p.beta !== nothing
        @test length(p.beta) == 2
    end

    @testset "dof counts profiled β" begin
        K = length(td.y)
        r0 = fit_mle(BipartiteVarianceStableModel, td.f, td.w, td.y;
            solver=ExactCholesky(optim_iters=5, polish=false), seed=1)
        rX = fit_mle(BipartiteVarianceStableModel, td.f, td.w, td.y;
            X=ones(K, 1),
            solver=ExactCholesky(optim_iters=5, polish=false), seed=1)
        @test dof(rX) == dof(r0) + 1
    end

    @testset "X dimension validation" begin
        K = length(td.y)
        @test_throws ArgumentError suffstats(BipartiteVarianceStableModel,
            td.f, td.w, td.y; X=ones(K+1, 1))
        @test_throws ArgumentError suffstats(BipartiteVarianceStableModel,
            td.f, td.w, td.y; X=ones(K, 0))
    end
end
