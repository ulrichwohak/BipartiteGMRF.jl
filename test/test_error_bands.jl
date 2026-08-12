# Banded within-firm error correlation (eta): the error covariance becomes
# sigma_eps^2 * R with R block-diagonal by firm, banded across each firm's
# spell-ordered observations. Everything flows through R-weighted sufficient
# statistics plus one likelihood constant, so the sharpest test is a DENSE
# reference: the exact-solver NLL must equal the closed-form Gaussian NLL with
# Sigma = V Q^-1 V' + sigma^2 R, exactly.

@testset "banded error correlation (eta)" begin
    # A small panel: firm 4 carries a co-CEO match (rows 8-9 share match 8) and
    # three spells; firms 1-3, 5 give singletons and short sequences.
    f = [1, 1, 1, 2, 2, 3, 4, 4, 4, 4, 5]
    w = [1, 2, 3, 2, 4, 5, 1, 6, 7, 3, 7]
    mid = [1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 10]
    rank = [1, 2, 3, 1, 2, 1, 1, 2, 2, 3, 1]
    y = [0.7, -0.3, 0.9, 0.2, -0.5, 1.1, -0.8, 0.4, 0.4, -0.2, 0.6]
    band = [0.4, 0.15]

    dense_R(firm_of, rank_of) = begin
        K = length(firm_of)
        R = Matrix{Float64}(I, K, K)
        for a in 1:K, b in 1:K
            a == b && continue
            firm_of[a] == firm_of[b] || continue
            d = abs(rank_of[a] - rank_of[b])
            1 <= d <= length(band) && (R[a, b] = band[d])
        end
        R
    end

    @testset "zero band reproduces the plain statistics" begin
        for m in (nothing, mid)
            ss0 = suffstats(BipartiteNormalizedModel, f, w, y;
                weighting = Weighting(observations = :raw), match_id = m,
                standardize = false,
                error_bands = (rank = rank, band = [0.0, 0.0]))
            ssp = suffstats(BipartiteNormalizedModel, f, w, y;
                weighting = Weighting(observations = :raw), match_id = m,
                standardize = false)
            @test ss0.K == ssp.K
            @test Matrix(ss0.design.VtV) ≈ Matrix(ssp.design.VtV) atol = 1e-12
            @test ss0.design.projected_y ≈ ssp.design.projected_y atol = 1e-12
            @test ss0.design.ydot ≈ ssp.design.ydot atol = 1e-12
            @test ss0.weights.log_weight_sum ≈ 0.0 atol = 1e-12
        end
    end

    @testset "dense likelihood reference, ungrouped" begin
        ss = suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw), standardize = false,
            error_bands = (rank = rank, band = band))
        model = BipartiteNormalizedModel(ss.A_prior; rho_limit = 0.99)
        rho, sa, sz, se = 0.25, 0.8, 0.5, 0.4
        params = [atanh(rho / 0.99), log(sa), log(sz), log(se)]
        obs_stats = BipartiteGMRF.objective_stats(model, ss, params)
        exact_nll = BipartiteGMRF.nll_exact_value(model, ss, params, obs_stats)

        K = length(y)
        n = ss.N_firms + ss.N_workers
        V = sparse(repeat(1:K, 2), vcat(f, ss.N_firms .+ w), ones(2K), K, n)
        Q = BipartiteGMRF.model_precision(model, rho, sa, sz)
        R = dense_R(f, rank)
        Sigma = Matrix(V * inv(Matrix(Q)) * transpose(V)) + se^2 .* R
        dense_nll = 0.5 * (logdet(Symmetric(Sigma)) + dot(y, Sigma \ y))
        @test exact_nll ≈ dense_nll atol = 1e-8 rtol = 1e-8
    end

    @testset "dense likelihood reference, match-grouped" begin
        ss = suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw), match_id = mid,
            standardize = false,
            error_bands = (rank = rank, band = band))
        model = BipartiteNormalizedModel(ss.A_prior; rho_limit = 0.99)
        rho, sa, sz, se = 0.3, 0.7, 0.6, 0.5
        params = [atanh(rho / 0.99), log(sa), log(sz), log(se)]
        obs_stats = BipartiteGMRF.objective_stats(model, ss, params)
        exact_nll = BipartiteGMRF.nll_exact_value(model, ss, params, obs_stats)

        # grouped observations by hand, first-appearance order of match ids
        mids = unique(mid)
        K = length(mids)
        n = ss.N_firms + ss.N_workers
        Vi = Int[]; Vj = Int[]; Vv = Float64[]
        yg = Float64[]; fg = Int[]; rg = Int[]
        for (s, m) in enumerate(mids)
            rows = findall(==(m), mid)
            ws = unique(w[rows])
            push!(yg, y[rows[1]]); push!(fg, f[rows[1]]); push!(rg, rank[rows[1]])
            push!(Vi, s); push!(Vj, f[rows[1]]); push!(Vv, 1.0)
            for wj in ws
                push!(Vi, s); push!(Vj, ss.N_firms + wj); push!(Vv, 1.0 / length(ws))
            end
        end
        V = sparse(Vi, Vj, Vv, K, n)
        Q = BipartiteGMRF.model_precision(model, rho, sa, sz)
        R = dense_R(fg, rg)
        Sigma = Matrix(V * inv(Matrix(Q)) * transpose(V)) + se^2 .* R
        dense_nll = 0.5 * (logdet(Symmetric(Sigma)) + dot(yg, Sigma \ yg))
        @test exact_nll ≈ dense_nll atol = 1e-8 rtol = 1e-8
    end

    @testset "mean structure composes: zero band reproduces plain mean stats" begin
        X = hcat(ones(length(y)), Float64.(f))
        ss0 = suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw), standardize = false, X = X,
            error_bands = (rank = rank, band = [0.0]))
        ssp = suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw), standardize = false, X = X)
        @test ss0.mean_stats.VtX ≈ ssp.mean_stats.VtX atol = 1e-12
        @test ss0.mean_stats.XtX ≈ ssp.mean_stats.XtX atol = 1e-12
        @test ss0.mean_stats.Xty ≈ ssp.mean_stats.Xty atol = 1e-12
    end

    @testset "fit_mle runs end to end with a band" begin
        result = fit_mle(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw), match_id = mid,
            error_bands = (rank = rank, band = band),
            solver = ExactCholesky(optim_iters = 40, polish = false))
        @test isfinite(result.nll)
        @test isfinite(result.rho)
    end

    @testset "guardrails" begin
        # non-PD block: strong band on a long consecutive run at one firm
        fL = fill(1, 6); wL = collect(1:6); rL = collect(1:6)
        yL = [0.3, -0.4, 0.1, 0.8, -0.2, 0.5]
        @test_throws ArgumentError suffstats(BipartiteNormalizedModel, fL, wL, yL;
            weighting = Weighting(observations = :raw), standardize = false,
            error_bands = (rank = rL, band = [0.9, 0.8]))
        # bands demand :raw
        @test_throws ArgumentError suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :edge),
            error_bands = (rank = rank, band = band))
        # rank length mismatch
        @test_throws ArgumentError suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw),
            error_bands = (rank = rank[1:3], band = band))
        # multi-firm match
        f2 = [1, 2]; w2 = [1, 1]; m2 = [1, 1]; r2 = [1, 1]; y2 = [0.5, 0.5]
        @test_throws ArgumentError suffstats(BipartiteNormalizedModel, f2, w2, y2;
            weighting = Weighting(observations = :raw), match_id = m2,
            error_bands = (rank = r2, band = band))
    end
end
