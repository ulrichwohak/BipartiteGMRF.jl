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

    @testset "profiling accepts borrowed solve buffers (issue #125)" begin
        ms = BipartiteGMRF.MeanStats(
            reshape([1.0, 1.0], 2, 1), reshape([2.0], 1, 1), [1.7], 1)
        factor = cholesky(Symmetric([2.0 0.0; 0.0 3.0]))
        scratch = zeros(2)
        fresh_solve(v) = factor \ v
        borrowed_solve(v) = copyto!(scratch, factor \ v)

        # Independent scalar GLS algebra: c = -0.3, G = 7/6.
        # A later solve must not overwrite the y solution before c is formed.
        expected_beta = [-9/35]
        expected_correction = 27/350
        for solve_M in (fresh_solve, borrowed_solve), _ in 1:2
            correction, beta = BipartiteGMRF.mean_profile_correction(
                ms, 1.0, [2.0, 3.0], solve_M)
            @test beta ≈ expected_beta atol=1e-13 rtol=1e-13
            @test correction ≈ expected_correction atol=1e-13 rtol=1e-13
        end
    end

    @testset "iid HutchSLQ mean correction agrees with dense GLS" begin
        # This supported iterative path really returns borrowed PCG storage.
        # Two controls exercise consecutive RHS solves within one profile.
        f = [1, 1, 2, 2, 3, 3, 4, 4, 4]
        w = [1, 2, 2, 3, 3, 4, 1, 4, 5]
        y = [0.3, 1.1, -0.2, 0.6, 1.5, -0.4, 0.8, -0.5, 0.2]
        X = hcat(ones(length(y)), [0.0, 1, 0, 1, 0, 1, 0, 1, 2])
        ss0 = suffstats(BipartiteNormalizedModel, f, w, y;
            weighting=Weighting(observations=:raw), standardize=false)
        ssX = suffstats(BipartiteNormalizedModel, f, w, y;
            weighting=Weighting(observations=:raw), standardize=false, X=X)
        model = BipartiteNormalizedModel(ssX.A_prior; rho_limit=0.8)
        rho, sa, sz, se = 0.3, 0.8, 0.6, 0.5
        params = [atanh(rho/0.8), log(sa), log(sz), log(se)]
        obs0 = BipartiteGMRF.objective_stats(model, ss0, params)
        obsX = BipartiteGMRF.objective_stats(model, ssX, params)
        @test obs0.design.VtV == obsX.design.VtV
        @test obs0.design.projected_y == obsX.design.projected_y

        nf, nw = 4, 5
        V = zeros(length(y), nf + nw)
        for i in eachindex(y)
            V[i, f[i]] = 1.0
            V[i, nf+w[i]] = 1.0
        end
        # Independently assemble normalized Q and observation-space GLS;
        # do not use mean statistics or the profiling helper as the oracle.
        A = Matrix(ssX.A_prior)
        W = A ./ sqrt.(vec(sum(A; dims=2)) * vec(sum(A; dims=1))')
        Q = [Matrix(Diagonal(fill(1/sa^2, nf))) -rho .* W ./ (sa*sz);
             -rho .* W' ./ (sa*sz) Matrix(Diagonal(fill(1/sz^2, nw)))]
        Omega = Symmetric(V * (Q \ V') + se^2 * I)
        c = X' * (Omega \ y)
        G = Symmetric(X' * (Omega \ X))
        expected_beta = G \ c
        expected_correction = dot(c, expected_beta)
        expected_nll0 = 0.5 * (logdet(Omega) + dot(y, Omega \ y))
        expected_nllX = expected_nll0 - 0.5 * expected_correction
        @test BipartiteGMRF.nll_exact_value(model, ss0, params, obs0) ≈
            expected_nll0 atol=1e-10 rtol=1e-10
        @test BipartiteGMRF.nll_exact_value(model, ssX, params, obsX) ≈
            expected_nllX atol=1e-10 rtol=1e-10

        solver = HutchSLQ(logdet_probes=8, lanczos_iters=nf+nw,
            cg_tol=1e-13, cg_maxiter=100, optim_iters=1)
        @test BipartiteGMRF.validate_capability(model, ssX, solver) === nothing
        cache0 = BipartiteGMRF.make_hutch_cache(model, ss0, solver)
        cacheX = BipartiteGMRF.make_hutch_cache(model, ssX, solver)
        for seed in (17, 42, 17)
            plain = BipartiteGMRF.nll_hutch_value(
                model, ss0, solver, params, obs0, cache0; seed=seed)
            profiled = BipartiteGMRF.nll_hutch_value(
                model, ssX, solver, params, obsX, cacheX; seed=seed)
            @test all(isfinite, (plain, profiled))
            @test max(plain, profiled) < BipartiteGMRF.BIG_NLL
            # Identical covariance parameters and probe streams cancel the
            # stochastic logdet error. This checks the actual objective's
            # mean correction without relying on a lucky Hutchinson seed.
            @test 2 * (plain - profiled) ≈ expected_correction atol=1e-10 rtol=1e-10
        end

        function borrowed_pcg(v)
            sol, ok, _, _ = BipartiteGMRF.pcg_solve!(cacheX.pcg, cacheX.mop, v;
                tol=solver.cg_tol, maxiter=solver.cg_maxiter, Mdiag=cacheX.Mdiag)
            @test ok
            return sol
        end
        correction, beta = BipartiteGMRF.mean_profile_correction(
            obsX.mean_stats, inv(se^2), obsX.design.projected_y, borrowed_pcg)
        @test beta ≈ expected_beta atol=1e-10 rtol=1e-10
        @test correction ≈ expected_correction atol=1e-10 rtol=1e-10
    end
end
