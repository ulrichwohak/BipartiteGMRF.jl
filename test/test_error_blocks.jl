@testset "free per-firm error blocks (error_blocks=:free)" begin
    f = [1, 1, 1, 2, 2]
    w = [1, 2, 3, 2, 4]
    y = [0.5, -0.3, 0.9, 0.2, -0.5]

    @testset "block discovery keeps observations at edge level" begin
        ss = suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw), standardize = false,
            error_blocks = :free, firm_group = f)
        fb = ss.free_blocks
        @test fb !== nothing
        @test ss.K == 5                      # five edge-level rows, not two firm means
        @test length(fb.sizes) == 2          # firm 1 and firm 2
        @test fb.sizes == [3, 2]
        @test fb.distinct == [3, 2]          # distinct managers per block
        @test fb.dof_blocks == 9             # 3*4/2 + 2*3/2
        @test fb.block_of == [1, 1, 1, 2, 2]
        @test size(fb.V) == (5, 6)           # (K, n_firms + n_workers)
        @test fb.y == y
    end

    @testset "duplicate manager row is rejected (rank-deficiency)" begin
        # firm 1's rows 1-2 share manager 1 → m_1 > d_1
        fdup = [1, 1, 1, 2]
        wdup = [1, 1, 2, 3]
        ydup = [0.5, -0.3, 0.9, 0.2]
        @test_throws ArgumentError suffstats(BipartiteNormalizedModel, fdup, wdup, ydup;
            weighting = Weighting(observations = :raw), standardize = false,
            error_blocks = :free, firm_group = fdup)
    end

    @testset "validation guardrails" begin
        R = sparse(1.0 * LinearAlgebra.I, length(y), length(y))
        # mutually exclusive with error_cov
        @test_throws ArgumentError suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw),
            error_blocks = :free, firm_group = f, error_cov = R)
        # mutually exclusive with error_groups
        @test_throws ArgumentError suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw),
            error_blocks = :free, firm_group = f, error_groups = f)
        # firm_group required
        @test_throws ArgumentError suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw), error_blocks = :free)
        # firm_group length mismatch
        @test_throws ArgumentError suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw),
            error_blocks = :free, firm_group = f[1:3])
        # requires :raw
        @test_throws ArgumentError suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :edge),
            error_blocks = :free, firm_group = f)
        # unknown mode
        @test_throws ArgumentError suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw),
            error_blocks = :structured, firm_group = f)
        # a firm_group value spanning multiple firms must be rejected
        @test_throws ArgumentError suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw),
            error_blocks = :free, firm_group = fill(1, length(f)))
    end
    @testset "dense reference: assembly, NLL, E-step" begin
        ss = suffstats(BipartiteVarianceStableModel, f, w, y;
            weighting = Weighting(observations = :raw), standardize = false,
            error_blocks = :free, firm_group = f)
        fb = ss.free_blocks
        nf, nw = ss.N_firms, ss.N_workers
        K = ss.K
        model = BipartiteVarianceStableModel(ss.A_prior; rho_limit = 0.99)
        ew = BipartiteGMRF.make_emfree_workspace(model, fb, nf, nw)

        Om1 = [1.2 0.4 0.1; 0.4 0.9 -0.2; 0.1 -0.2 1.5]
        Om2 = [1.0 0.3; 0.3 0.8]
        Ωs = [Om1, Om2]
        Om = zeros(K, K); Om[1:3, 1:3] = Om1; Om[4:5, 4:5] = Om2
        A = Matrix(fb.V)

        P, b = BipartiteGMRF.assemble_freeblock_precision(fb, Ωs, nf, nw)
        @test Matrix(P) ≈ A' * inv(Om) * A atol = 1e-12
        @test b ≈ vec(A' * (inv(Om) * fb.y)) atol = 1e-12

        rho, sa, sz = 0.3, 0.7, 0.6
        Q = Matrix(BipartiteGMRF.model_precision(model, rho, sa, sz))
        V = A * inv(Q) * A' + Om
        nll_em = BipartiteGMRF.emfree_nll(model, fb, ew, Ωs, rho, sa, sz, nf, nw)
        nll_dense = 0.5 * (logdet(Symmetric(V)) + dot(fb.y, V \ fb.y))
        @test nll_em ≈ nll_dense atol = 1e-8 rtol = 1e-8

        alpha, Seps, _ = BipartiteGMRF.emfree_e_step(model, fb, ew, Ωs, rho, sa, sz, nf, nw)
        M = Q + A' * inv(Om) * A
        M_inv = inv(M)
        r = fb.y - A * alpha
        for (i, idx) in enumerate([[1, 2, 3], [4, 5]])
            Ai = fb.V[idx, :]
            @test Seps[i] ≈ r[idx] * r[idx]' + Ai * M_inv * Ai' atol = 1e-12
        end
    end

    @testset "reduction to iid: Ω = σ²I reproduces V'V" begin
        ss = suffstats(BipartiteVarianceStableModel, f, w, y;
            weighting = Weighting(observations = :raw), standardize = false,
            error_blocks = :free, firm_group = f)
        fb = ss.free_blocks
        nf, nw = ss.N_firms, ss.N_workers
        σ² = 0.7
        Ωs = [σ² .* Matrix{Float64}(LinearAlgebra.I, m, m) for m in fb.sizes]
        P, b = BipartiteGMRF.assemble_freeblock_precision(fb, Ωs, nf, nw)
        iid = ss.design.VtV
        @test Matrix(P) ≈ (1 / σ²) .* Matrix(iid) atol = 1e-12
        @test b ≈ (1 / σ²) .* ss.design.projected_y atol = 1e-12
    end
    @testset "spectral floor clamps a near-singular block" begin
        # one slightly-negative eigenvalue → must be floored up to λ_lo
        S = [-0.01 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 2.0]
        Fb = BipartiteGMRF._floor_spectrum(S, 0.1)
        @test issymmetric(Fb)
        @test eigen(Symmetric(Fb)).values[1] ≈ 0.1 atol = 1e-12
        # a block already above the floor is returned untouched
        S2 = [1.0 0.2 0.0; 0.2 1.0 0.2; 0.0 0.2 1.0]
        @test BipartiteGMRF._floor_spectrum(S2, 0.1) == S2
    end

    @testset "non-convergence is warned and surfaced" begin
        res = @test_logs (:warn, r"did not converge") match_mode=:any fit_mle(
            BipartiteVarianceStableModel, f, w, y;
            weighting = Weighting(observations = :raw), standardize = false,
            error_blocks = :free, firm_group = f,
            solver = EMFreeBlocks(max_iter = 2, ftol = 1e-12))
        @test res.converged == false
    end

    @testset "EM NLL is monotone non-increasing" begin
        ss = suffstats(BipartiteVarianceStableModel, f, w, y;
            weighting = Weighting(observations = :raw), standardize = false,
            error_blocks = :free, firm_group = f)
        model = BipartiteVarianceStableModel(ss.A_prior; rho_limit = 0.99)
        fit = BipartiteGMRF.optimize_emfree(model, ss,
            EMFreeBlocks(max_iter = 30, ftol = 1e-10, eig_floor = 1e-3); verbose = false)
        @test length(fit.nll_trace) == 31
        @test all(diff(fit.nll_trace) .<= 1e-9)
    end
end