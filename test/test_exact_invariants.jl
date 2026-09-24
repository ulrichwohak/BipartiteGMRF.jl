@testset "ExactCholesky sparse-pattern invariants" begin
    BG = BipartiteGMRF
    GM = BG.GaussianMarkovRandomFields

    @testset "checked sparse alignment" begin
        target = sparse([1, 3, 5, 2, 4], [1, 1, 1, 2, 2], ones(5), 5, 2)
        A = sparse([3, 2], [1, 2], [2.0, -4.0], 5, 2)
        aligned = BG._align_to_pattern(A, target)
        @test aligned.colptr == target.colptr
        @test rowvals(aligned) == rowvals(target)
        @test Matrix(aligned) == Matrix(A)
        @test nonzeros(target) == ones(5)
        @test nnz(aligned) == nnz(target)

        # Accumulation must retain zero-valued positions after cancellation.
        vals = zeros(nnz(target))
        BG._add_to_pattern!(vals, A, target)
        @test vals == nonzeros(aligned)
        BG._add_to_pattern!(vals, A, target, -1.0)
        @test iszero(vals)
        BG._add_to_pattern!(vals, A, target, 2.0)
        @test vals == 2.0 .* nonzeros(aligned)

        # Unsupported rows must not disappear at any point in a CSC column.
        for (source_rows, target_rows) in (
            ([1, 3], [3]),        # before the first target position
            ([1, 2, 3], [1, 3]),  # between target positions
            ([1, 3], [1]),        # after the final target position
            ([1, 2], [1, 3]),     # equal nnz, different positions
        )
            source = sparse(source_rows, ones(Int, length(source_rows)),
                ones(length(source_rows)), 3, 1)
            pattern = sparse(target_rows, ones(Int, length(target_rows)),
                ones(length(target_rows)), 3, 1)
            @test_throws ArgumentError BG._align_to_pattern(source, pattern)
            @test_throws ArgumentError BG._add_to_pattern!(zeros(nnz(pattern)), source, pattern)
        end

        # Structural support, rather than the current value, is the contract.
        explicit_zero = sparse([2], [1], [0.0], 3, 1)
        empty_pattern = spzeros(3, 1)
        @test nnz(explicit_zero) == 1
        @test_throws ArgumentError BG._align_to_pattern(explicit_zero, empty_pattern)
        @test_throws ArgumentError BG._add_to_pattern!(Float64[], explicit_zero, empty_pattern)
        @test_throws DimensionMismatch BG._align_to_pattern(spzeros(2, 1), spzeros(3, 1))
        @test_throws DimensionMismatch BG._align_to_pattern(spzeros(3, 2), spzeros(3, 1))
        @test_throws DimensionMismatch BG._add_to_pattern!(zeros(4), A, target)
    end

    @testset "incompatible caches are errors, not infeasible parameters" begin
        f, w, y = [1, 1, 1], [1, 2, 1], [0.3, 0.3, 1.2]
        ss = suffstats(BipartiteNormalizedModel, f, w, y;
            match_id=[1, 1, 2], edge_index=[1, 1, 2], error_eta=:estimate,
            weighting=Weighting(observations=:raw), standardize=false)
        model = BipartiteNormalizedModel(ss.A_prior; rho_limit=0.8)
        p = [atanh(0.2 / 0.8), log(0.8), log(0.6), log(0.5), atanh(0.51)]
        obs = BG.objective_stats(model, ss, p)
        cache = BG.make_exact_workspace(model, ss)

        missing_Q = BG.ExactWorkspace(GM.GMRFWorkspace(spdiagm(0 => ones(3))),
            GM.GMRFWorkspace(copy(cache.ws_M.Q)))
        @test_throws ArgumentError BG.nll_exact_value(model, ss, p, obs, missing_Q)

        # Q has no worker-worker edge, but the AR(1) posterior needs one.
        missing_M = BG.ExactWorkspace(GM.GMRFWorkspace(copy(cache.ws_Q.Q)),
            GM.GMRFWorkspace(copy(cache.ws_Q.Q)))
        @test_throws ArgumentError BG.nll_exact_value(model, ss, p, obs, missing_M)

        wrong_dimension = BG.ExactWorkspace(GM.GMRFWorkspace(spdiagm(0 => ones(2))),
            GM.GMRFWorkspace(copy(cache.ws_M.Q)))
        @test_throws DimensionMismatch BG.nll_exact_value(model, ss, p, obs, wrong_dimension)

        # At eta=0.5 this fixture has seven current nonzeros. A different
        # seven-entry pattern must still be rejected: nnz alone is insufficient.
        p_half = [p[1:4]; atanh(0.5)]
        obs_half = BG.objective_stats(model, ss, p_half)
        Q = BG.model_precision(model, 0.2, 0.8, 0.6)
        M = Q + 4.0 .* obs_half.design.VtV
        wrong_pattern = sparse([1, 2, 3, 1, 2, 2, 3], [1, 2, 3, 2, 1, 3, 2],
            [1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1], 3, 3)
        @test nnz(wrong_pattern) == nnz(M)
        incompatible = BG.ExactWorkspace(GM.GMRFWorkspace(copy(cache.ws_Q.Q)),
            GM.GMRFWorkspace(wrong_pattern))
        @test_throws ArgumentError BG.nll_exact_value(model, ss, p_half, obs_half, incompatible)
    end

    @testset "numerical infeasibility preserves subsequent cache reuse" begin
        # K_3,3 has an indefinite VS precision at rho=0.8, despite rho<1.
        # Its non-backtracking feasibility threshold is 1/(3-1)=0.5.
        f, w = repeat(1:3; inner=3), repeat(1:3; outer=3)
        y = [0.1, -0.2, 0.4, 0.3, 0.8, -0.1, -0.4, 0.2, 0.7]
        ss = suffstats(BipartiteVarianceStableModel, f, w, y;
            weighting=Weighting(observations=:raw), standardize=false)
        model = @test_warn "contains a cycle" BipartiteVarianceStableModel(ss.A_prior;
            rho_limit=0.99, strict_forest=false)
        cache = BG.make_exact_workspace(model, ss)
        p_bad = [atanh(0.8 / 0.99), log(0.8), log(0.6), log(0.5)]
        Q_bad = BG.model_precision(model, 0.8, 0.8, 0.6)
        @test !isposdef(Symmetric(Matrix(Q_bad)))
        obs_bad = BG.objective_stats(model, ss, p_bad)
        @test BG.nll_exact_value(model, ss, p_bad, obs_bad, cache) == BG.BIG_NLL

        p_good = [atanh(0.2 / 0.99), p_bad[2:4]...]
        obs_good = BG.objective_stats(model, ss, p_good)
        Q_good = BG.model_precision(model, 0.2, 0.8, 0.6)
        V = zeros(length(y), 6)
        for k in eachindex(y)
            V[k, f[k]] = 1.0
            V[k, 3 + w[k]] = 1.0
        end
        Sigma = V * (Matrix(Q_good) \ V') + 0.5^2 * I
        dense = 0.5 * (logdet(Symmetric(Sigma)) + dot(y, Sigma \ y))
        @test BG.nll_exact_value(model, ss, p_good, obs_good, cache) ≈ dense atol=1e-10
        @test issuccess(cache.ws_Q.backend.factor)
        @test issuccess(cache.ws_M.backend.factor)
    end

    @testset "polishing does not conceal structural failures" begin
        result = BG.optimize(x -> sum(abs2, x), [1.0], BG.NelderMead(),
            BG.Options(iterations=2))
        solver = ExactCholesky(optim_iters=2, polish=true)
        for error in (ArgumentError("incompatible sparse pattern"),
                      DimensionMismatch("incompatible cache dimensions"),
                      InterruptException())
            objective = _ -> throw(error)
            @test_throws typeof(error) BG.polish(solver, objective, result, false)
        end
    end

    @testset "independent reference is feasible on dense cyclic VS graphs" begin
        # The old rho_ref=0.1 is already outside the feasible interval for
        # K_12,12. Structural support lets initialization safely use rho=0.
        f, w = repeat(1:12; inner=12), repeat(1:12; outer=12)
        y = sin.(collect(1:length(f)))
        ss = suffstats(BipartiteVarianceStableModel, f, w, y;
            weighting=Weighting(observations=:raw), standardize=false)
        model = @test_warn "contains a cycle" BipartiteVarianceStableModel(ss.A_prior;
            rho_limit=0.99, strict_forest=false)
        @test !isposdef(Symmetric(Matrix(BG.model_precision(model, 0.1, 1.0, 1.0))))
        cache = BG.make_exact_workspace(model, ss)
        @test issuccess(cache.ws_Q.backend.factor)
        @test issuccess(cache.ws_M.backend.factor)
        @test nnz(cache.ws_Q.Q) == 24 + 2 * 144
        params = [atanh(0.02 / 0.99), log(0.8), log(0.6), log(0.5)]
        obs = BG.objective_stats(model, ss, params)
        @test BG.nll_exact_value(model, ss, params, obs, cache) < BG.BIG_NLL
    end
end
