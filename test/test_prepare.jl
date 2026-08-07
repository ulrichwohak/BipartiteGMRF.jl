@testset "prepare" begin
    ss = suffstats_synthetic()
    @test ss.N_firms == 3
    @test ss.N_workers == 4
    @test ss.K == 8
    @test ss.metadata.unique_edges == 8

    @test BipartiteGMRF.rho_limit(BipartiteNormalizedModel(sparse(ones(2,2)); rho_limit=0.4)) == 0.4
    @test BipartiteGMRF.rho_limit(BipartiteUnnormalizedModel(sparse(ones(2,2)); rho_limit=0.4)) == 0.4
    @test BipartiteGMRF.rho_limit(BipartiteSpectralModel(sparse(ones(2,2)); rho_limit=0.4)) == 0.4
    @test BipartiteGMRF.rho_limit(BipartiteVarianceStableModel(sparse([1.0], [1.0], [1.0], 1, 1); rho_limit=0.4)) == 0.4
    @test_throws ArgumentError BipartiteNormalizedModel(sparse(ones(2,2)); rho_limit=1.0)

    # The estimator does not manipulate data: bad input is rejected, not fixed.
    d = synthetic_data()
    @test_throws ArgumentError suffstats(BipartiteNormalizedModel, d.f, d.w, d.y[1:4])
    @test_throws ArgumentError suffstats(BipartiteNormalizedModel, d.f, d.w, d.y; n_firms=2)
    @test_throws ArgumentError suffstats(BipartiteNormalizedModel, Int[], Int[], Float64[])
    # All-NaN is rejected (no finite outcomes).
    @test_throws ArgumentError suffstats(BipartiteNormalizedModel, d.f, d.w, fill(NaN, 8))
    # A node index with no observations AND no graph-only edge produces a
    # zero-degree node at model construction time.
    @test_throws ArgumentError fit_mle(BipartiteNormalizedModel, d.f, d.w, d.y; n_firms=4)

    edge_ss = suffstats_repeated(; weighting=Weighting(observations=:edge))
    @test edge_ss.K == 8
    @test edge_ss.personyear_rows == 10

    effective_ss = suffstats_repeated(;
        weighting=Weighting(observations=:effective, rho_eps=0.5),
    )
    @test effective_ss.rho_eps_likelihood == 0.5
    @test effective_ss.weights.effective_weight_sum < effective_ss.personyear_rows

    @test_warn "variance-stable model no longer guarantees" BipartiteVarianceStableModel(
        suffstats_synthetic(BipartiteVarianceStableModel).A_prior,
    )
    @test_throws ArgumentError BipartiteVarianceStableModel(
        suffstats_synthetic(BipartiteVarianceStableModel).A_prior;
        strict_forest=true,
    )

    @testset "collapse_edges" begin
        r = repeated_data()
        edges = BipartiteGMRF.collapse_edges(r.f, r.w, r.y)
        @test length(edges.f) == 8
        @test sum(edges.T) == 10
        # Edge (1,1) appears in rows 1 and 2.
        j = findfirst(i -> edges.f[i] == 1 && edges.w[i] == 1, eachindex(edges.f))
        @test edges.T[j] == 2
        @test edges.y_mean[j] ≈ (1.2 + 1.1) / 2
        @test edges.ssw[j] ≈ (1.2 - 1.15)^2 + (1.1 - 1.15)^2
    end

    @testset "replace_stats" begin
        ss2 = BipartiteGMRF.replace_stats(ss; K=99)
        @test ss2.K == 99
        @test ss2.N_firms == ss.N_firms
        @test ss2.design === ss.design
    end

    @testset "graph-only edges" begin
        d = synthetic_data()
        ss_ref = suffstats(BipartiteNormalizedModel, d.f, d.w, d.y)

        # ── regression: all-finite y reproduces existing stats ──
        @test ss_ref.K == 8
        @test ss_ref.metadata.graph_only_rows == 0
        @test ss_ref.metadata.graph_only_edges == 0

        # ── NaN outcome → graph-only edge ──
        # Add a graph-only edge (firm 3, worker 1) — not in original observations.
        f_ext = [d.f; 3]
        w_ext = [d.w; 1]
        y_ext = [d.y; NaN]
        ss_ext = suffstats(BipartiteNormalizedModel, f_ext, w_ext, y_ext)

        # A_prior gains the extra edge
        @test nnz(ss_ext.A_prior) == nnz(ss_ref.A_prior) + 1
        @test ss_ext.A_prior[3, 1] == 1.0  # new edge present

        # Likelihood inputs are identical (graph-only edge excluded)
        @test ss_ext.design.VtV == ss_ref.design.VtV
        @test ss_ext.design.projected_y == ss_ref.design.projected_y
        @test ss_ext.design.ydot ≈ ss_ref.design.ydot
        @test ss_ext.K == ss_ref.K

        # Metadata tracks graph-only
        @test ss_ext.metadata.graph_only_rows == 1
        @test ss_ext.metadata.graph_only_edges == 1

        # ── NaN on an existing edge: graph stays the same ──
        f_dup = [d.f; 1]
        w_dup = [d.w; 1]
        y_dup = [d.y; NaN]
        ss_dup = suffstats(BipartiteNormalizedModel, f_dup, w_dup, y_dup)
        @test nnz(ss_dup.A_prior) == nnz(ss_ref.A_prior)  # no new edge
        @test ss_dup.design.VtV == ss_ref.design.VtV        # likelihood unchanged
        @test ss_dup.metadata.graph_only_rows == 1
        @test ss_dup.metadata.graph_only_edges == 0          # edge already existed

        # ── Node only reachable through graph-only edges ──
        f_new = [d.f; 4]
        w_new = [d.w; 1]
        y_new = [d.y; NaN]
        ss_new = suffstats(BipartiteNormalizedModel, f_new, w_new, y_new;
            n_firms=4)
        @test ss_new.N_firms == 4
        @test ss_new.A_prior[4, 1] == 1.0
        # Node 4 has zero observation count but nonzero prior degree
        @test ss_new.design.FF[4, 4] == 0.0
        @test sum(ss_new.A_prior[4, :]) == 1.0

        # ── Fitting succeeds with graph-only edges ──
        result = fit_mle(BipartiteNormalizedModel, f_ext, w_ext, y_ext;
            solver=ExactCholesky(optim_iters=5, polish=false), seed=1)
        @test isfinite(result.nll)
        @test isfinite(result.rho)
        @test result.sigma_a > 0
        @test nnz(result.model.graph.A) == nnz(ss_ref.A_prior) + 1

        # ── Decomposition and covariance work ──
        vd = decompose(result; kind=:model, probes=20, seed=1)
        @test isfinite(vd.V_firm)
        @test isfinite(vd.V_worker)
        op = covariance(result; kind=:model)
        blk = cov_block(op; firms=[1, 3], workers=[1])
        @test size(blk.matrix) == (3, 3)
    end

    @testset "match-grouped observations" begin
        # ── regression: unique match_id per edge reproduces existing stats ──
        d = synthetic_data()
        ss_ref = suffstats(BipartiteNormalizedModel, d.f, d.w, d.y)
        ss_mid = suffstats(BipartiteNormalizedModel, d.f, d.w, d.y;
            match_id=collect(1:length(d.y)))
        @test ss_mid.design.VtV ≈ ss_ref.design.VtV
        @test ss_mid.design.projected_y ≈ ss_ref.design.projected_y
        @test ss_mid.design.ydot ≈ ss_ref.design.ydot
        @test ss_mid.K == ss_ref.K

        # ── two workers, one firm: off-diagonal in WW ──
        # match 1: (firm=1, worker=1) + (firm=1, worker=2), y=1.0
        # match 2: (firm=1, worker=1), y=0.5
        f2 = [1, 1, 1]
        w2 = [1, 2, 1]
        y2 = [1.0, 1.0, 0.5]
        mid2 = [1, 1, 2]
        ss2 = suffstats(BipartiteNormalizedModel, f2, w2, y2;
            match_id=mid2, standardize=false)
        @test ss2.K == 2  # two matches
        # Match 1: F=1, M=2 → FF[1,1]+=1, WW[1,1]+=1/4, WW[1,2]+=1/4,
        #   WW[2,1]+=1/4, WW[2,2]+=1/4, A[1,1]+=1/2, A[1,2]+=1/2
        # Match 2: F=1, M=1 → FF[1,1]+=1, WW[1,1]+=1, A[1,1]+=1
        @test ss2.design.FF[1, 1] ≈ 2.0           # 1 + 1
        @test ss2.design.WW[1, 1] ≈ 1.25           # 1/4 + 1
        @test ss2.design.WW[1, 2] ≈ 0.25           # off-diagonal!
        @test ss2.design.WW[2, 1] ≈ 0.25
        @test ss2.design.WW[2, 2] ≈ 0.25
        @test ss2.design.A_obs[1, 1] ≈ 1.5         # 1/2 + 1
        @test ss2.design.A_obs[1, 2] ≈ 0.5         # 1/2
        # V'y: firm: 1/1*1.0 + 1/1*0.5 = 1.5; w1: 1/2*1.0 + 1/1*0.5 = 1.0; w2: 1/2*1.0 = 0.5
        @test ss2.design.projected_y ≈ [1.5, 1.0, 0.5]
        @test ss2.design.ydot ≈ 1.0^2 + 0.5^2

        # ── two firms, one worker: off-diagonal in FF ──
        f3 = [1, 2, 1]
        w3 = [1, 1, 2]
        y3 = [1.0, 1.0, 0.5]
        mid3 = [1, 1, 2]
        ss3 = suffstats(BipartiteNormalizedModel, f3, w3, y3;
            match_id=mid3, standardize=false)
        # Match 1: F=2, M=1 → FF[i,i']+=1/4, WW[1,1]+=1, A[1,1]+=1/2, A[2,1]+=1/2
        # Match 2: F=1, M=1 → FF[1,1]+=1, WW[2,2]+=1, A[1,2]+=1
        @test ss3.design.FF[1, 1] ≈ 1.25          # 1/4 + 1
        @test ss3.design.FF[1, 2] ≈ 0.25          # off-diagonal!
        @test ss3.design.FF[2, 1] ≈ 0.25
        @test ss3.design.WW[1, 1] ≈ 1.0           # 1/1^2 from match 1

        # ── re-hire: same (f,w) pair, two different match ids ──
        f4 = [1, 1]
        w4 = [1, 1]
        y4 = [1.0, 2.0]
        mid4 = [1, 2]
        ss4 = suffstats(BipartiteNormalizedModel, f4, w4, y4;
            match_id=mid4, standardize=false)
        @test ss4.K == 2  # two matches, one edge

        # ── inconsistent outcomes within match → error ──
        @test_throws ArgumentError suffstats(BipartiteNormalizedModel,
            [1, 2], [1, 2], [1.0, 2.0]; match_id=[1, 1])

        # ── mixed finite/NaN within match → error ──
        @test_throws ArgumentError suffstats(BipartiteNormalizedModel,
            [1, 2], [1, 2], [1.0, NaN]; match_id=[1, 1])

        # ── match_id with non-raw weighting → error ──
        @test_throws ArgumentError suffstats(BipartiteNormalizedModel,
            [1, 2], [1, 2], [1.0, 1.0];
            match_id=[1, 1], weighting=Weighting(observations=:edge))

        # ── standardize=true: match-weighted centering ──
        # Multi-edge match must not bias the mean toward its outcome.
        f_std = [1, 1, 2, 2]
        w_std = [1, 2, 1, 2]
        y_std_val = [3.0, 3.0, 1.0, 2.0]
        mid_std = [1, 1, 2, 3]
        ss_std = suffstats(BipartiteNormalizedModel, f_std, w_std, y_std_val;
            match_id=mid_std, standardize=true)
        # y_mean must be match-weighted (3+1+2)/3 = 2.0, not edge-weighted (3+3+1+2)/4 = 2.25
        @test ss_std.y_mean ≈ 2.0
        @test ss_std.y_std ≈ 1.0
        # Centered match outcomes [1, -1, 0] sum to zero through projections
        @test sum(ss_std.design.projected_y) ≈ 0.0 atol=1e-14

        # ── graph_only_edges under match grouping ──
        # Match 1 has 2 firms × 2 workers: A_obs gets 4 nonzeros from 2 edges.
        # graph_only_edges must count from the edge set, not nnz(A_obs).
        f_go = [1, 2, 3, 3]
        w_go = [1, 2, 1, 2]
        y_go = [1.0, 1.0, NaN, 2.0]
        mid_go = [1, 1, 2, 3]  # match 2 is graph-only
        ss_go = suffstats(BipartiteNormalizedModel, f_go, w_go, y_go;
            match_id=mid_go)
        @test ss_go.metadata.graph_only_edges == 1
        @test ss_go.metadata.graph_only_rows == 1

        # ── end-to-end fit + decompose ──
        f5 = [1, 2, 1, 2, 3]
        w5 = [1, 1, 2, 2, 2]
        y5 = [1.0, 1.0, 0.5, 0.5, 0.8]
        mid5 = [1, 1, 2, 2, 3]
        result = fit_mle(BipartiteNormalizedModel, f5, w5, y5;
            match_id=mid5,
            solver=ExactCholesky(optim_iters=5, polish=false), seed=1)
        @test isfinite(result.nll)
        @test result.stats.K == 3
        vd = decompose(result; kind=:model, probes=20, seed=1)
        @test isfinite(vd.V_total)
        vd_f = decompose(result; kind=:fitted, probes=20, seed=1)
        @test isfinite(vd_f.V_total)
    end
end
