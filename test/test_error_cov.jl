# General correlated errors: the error covariance becomes sigma_eps^2 * R,
# where R is a sparse symmetric matrix over observations whose connected
# blocks are arbitrary PD matrices (any correlation pattern, any
# heteroskedasticity); R's scale is pinned internally by tr(R) = K.
# Everything flows through R-weighted sufficient statistics plus one
# likelihood constant, so the sharpest test is a DENSE reference: the
# exact-solver NLL must equal the closed-form Gaussian NLL with
# Sigma = V Q^-1 V' + sigma^2 R_normalized, exactly.

@testset "correlated errors (error_cov)" begin
    # A small panel: firm 4 carries a co-CEO match (rows 8-9 share match 8);
    # firms 1-3, 5 give singletons and short sequences.
    f = [1, 1, 1, 2, 2, 3, 4, 4, 4, 4, 5]
    w = [1, 2, 3, 2, 4, 5, 1, 6, 7, 3, 7]
    mid = [1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 10]
    y = [0.7, -0.3, 0.9, 0.2, -0.5, 1.1, -0.8, 0.4, 0.4, -0.2, 0.6]
    K_in = length(y)

    # Arbitrary PD blocks, deliberately non-stationary and heteroskedastic:
    # firm 1 (rows 1-3), firm 2 (rows 4-5), firm 4 (rows 7-10); rows 6 and 11
    # are singletons, one with a non-unit variance.
    R_in = zeros(K_in, K_in)
    R_in[1:3, 1:3] = [1.2 0.5 0.2; 0.5 0.8 -0.1; 0.2 -0.1 1.5]
    R_in[4:5, 4:5] = [1.0 0.3; 0.3 0.7]
    R_in[6, 6] = 2.0
    R_in[7:10, 7:10] = [1.1 0.4 0.2 0.1; 0.4 0.9 0.3 0.0; 0.2 0.3 1.3 -0.2; 0.1 0.0 -0.2 0.6]
    R_in[11, 11] = 1.0
    R_sp = sparse(R_in)

    normalized(R::AbstractMatrix) = R .* (size(R, 1) / sum(R[i, i] for i in 1:size(R, 1)))

    @testset "identity error_cov reproduces the plain statistics" begin
        for m in (nothing, mid)
            ss0 = suffstats(BipartiteNormalizedModel, f, w, y;
                weighting = Weighting(observations = :raw), match_id = m,
                standardize = false,
                error_cov = sparse(1.0 * LinearAlgebra.I, K_in, K_in))
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

    @testset "scale of error_cov is irrelevant (tr(R) = K normalization)" begin
        ss1 = suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw), standardize = false,
            error_cov = R_sp)
        ss2 = suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw), standardize = false,
            error_cov = 3.7 .* R_sp)
        @test Matrix(ss1.design.VtV) ≈ Matrix(ss2.design.VtV) atol = 1e-12
        @test ss1.design.projected_y ≈ ss2.design.projected_y atol = 1e-12
        @test ss1.design.ydot ≈ ss2.design.ydot atol = 1e-12
        @test ss1.weights.log_weight_sum ≈ ss2.weights.log_weight_sum atol = 1e-12
    end

    @testset "dense likelihood reference, ungrouped" begin
        ss = suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw), standardize = false,
            error_cov = R_sp)
        model = BipartiteNormalizedModel(ss.A_prior; rho_limit = 0.99)
        rho, sa, sz, se = 0.25, 0.8, 0.5, 0.4
        params = [atanh(rho / 0.99), log(sa), log(sz), log(se)]
        obs_stats = BipartiteGMRF.objective_stats(model, ss, params)
        exact_nll = BipartiteGMRF.nll_exact_value(model, ss, params, obs_stats)

        n = ss.N_firms + ss.N_workers
        V = sparse(repeat(1:K_in, 2), vcat(f, ss.N_firms .+ w), ones(2K_in), K_in, n)
        Q = BipartiteGMRF.model_precision(model, rho, sa, sz)
        Sigma = Matrix(V * inv(Matrix(Q)) * transpose(V)) + se^2 .* normalized(R_in)
        dense_nll = 0.5 * (logdet(Symmetric(Sigma)) + dot(y, Sigma \ y))
        @test exact_nll ≈ dense_nll atol = 1e-8 rtol = 1e-8
    end

    @testset "dense likelihood reference, match-grouped" begin
        ss = suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw), match_id = mid,
            standardize = false,
            error_cov = R_sp)
        model = BipartiteNormalizedModel(ss.A_prior; rho_limit = 0.99)
        rho, sa, sz, se = 0.3, 0.7, 0.6, 0.5
        params = [atanh(rho / 0.99), log(sa), log(sz), log(se)]
        obs_stats = BipartiteGMRF.objective_stats(model, ss, params)
        exact_nll = BipartiteGMRF.nll_exact_value(model, ss, params, obs_stats)

        # grouped observations by hand, first-appearance order of match ids;
        # R is read at each match's first input row
        mids = unique(mid)
        K = length(mids)
        n = ss.N_firms + ss.N_workers
        Vi = Int[]; Vj = Int[]; Vv = Float64[]
        yg = Float64[]; srcg = Int[]
        for (s, m) in enumerate(mids)
            rows = findall(==(m), mid)
            fs = unique(f[rows]); ws = unique(w[rows])
            push!(yg, y[rows[1]]); push!(srcg, rows[1])
            for fj in fs
                push!(Vi, s); push!(Vj, fj); push!(Vv, 1.0 / length(fs))
            end
            for wj in ws
                push!(Vi, s); push!(Vj, ss.N_firms + wj); push!(Vv, 1.0 / length(ws))
            end
        end
        V = sparse(Vi, Vj, Vv, K, n)
        Q = BipartiteGMRF.model_precision(model, rho, sa, sz)
        Rg = normalized(R_in[srcg, srcg])
        Sigma = Matrix(V * inv(Matrix(Q)) * transpose(V)) + se^2 .* Rg
        dense_nll = 0.5 * (logdet(Symmetric(Sigma)) + dot(yg, Sigma \ yg))
        @test exact_nll ≈ dense_nll atol = 1e-8 rtol = 1e-8
    end

    @testset "mean structure composes: identity error_cov reproduces plain mean stats" begin
        X = hcat(ones(length(y)), Float64.(f))
        ss0 = suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw), standardize = false, X = X,
            error_cov = sparse(1.0 * LinearAlgebra.I, K_in, K_in))
        ssp = suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw), standardize = false, X = X)
        @test ss0.mean_stats.VtX ≈ ssp.mean_stats.VtX atol = 1e-12
        @test ss0.mean_stats.XtX ≈ ssp.mean_stats.XtX atol = 1e-12
        @test ss0.mean_stats.Xty ≈ ssp.mean_stats.Xty atol = 1e-12
    end

    @testset "fit_mle runs end to end with error_cov" begin
        result = fit_mle(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw), match_id = mid,
            error_cov = R_sp,
            solver = ExactCholesky(optim_iters = 40, polish = false))
        @test isfinite(result.nll)
        @test isfinite(result.rho)
    end

    @testset "guardrails" begin
        # a block that is not PD
        R_bad = sparse(1.0 * LinearAlgebra.I, K_in, K_in)
        R_bad[1, 2] = R_bad[2, 1] = 0.9
        R_bad[2, 2] = 0.8
        @test_throws ArgumentError suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw), standardize = false,
            error_cov = R_bad)
        # error_cov demands :raw
        @test_throws ArgumentError suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :edge), error_cov = R_sp)
        # wrong size
        @test_throws ArgumentError suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw),
            error_cov = R_sp[1:3, 1:3])
        # asymmetric
        R_asym = copy(R_sp); R_asym[1, 2] = 0.9
        @test_throws ArgumentError suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw), error_cov = R_asym)
    end
end
