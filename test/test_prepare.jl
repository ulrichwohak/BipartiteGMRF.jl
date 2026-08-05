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
    @test_throws ArgumentError suffstats(BipartiteNormalizedModel, d.f, d.w,
        [1.0, NaN, d.y[3:end]...])
    @test_throws ArgumentError suffstats(BipartiteNormalizedModel, d.f, d.w, d.y[1:4])
    @test_throws ArgumentError suffstats(BipartiteNormalizedModel, d.f, d.w, d.y; n_firms=2)
    @test_throws ArgumentError suffstats(BipartiteNormalizedModel, Int[], Int[], Float64[])
    # A node index with no observations produces a zero-degree node.
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
end
