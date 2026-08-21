# AR(1) within-firm error correlation: the error covariance is sigma_eps^2 * R(eta)
# with R(eta) block-diagonal by firm and block [eta^|k-l|], where k, l are the
# caller-supplied edge_index values (1..m_i within each firm). eta is either fixed
# in (-1, 1) or estimated jointly. The sharpest test is a dense reference: the
# exact-solver NLL must equal the closed-form Gaussian NLL with
# Sigma = V Q^-1 V' + sigma^2 R(eta), exactly.

@testset "AR(1) within-firm errors (error_eta)" begin
    f = [1, 1, 1, 2, 2, 3, 4, 4, 4, 4, 5]
    w = [1, 2, 3, 2, 4, 5, 1, 6, 7, 3, 7]
    y = [0.7, -0.3, 0.9, 0.2, -0.5, 1.1, -0.8, 0.4, 0.4, -0.2, 0.6]
    K_in = length(y)
    # within-firm edge order: firm sizes 3, 2, 1, 4, 1.
    eidx = [1, 2, 3, 1, 2, 1, 1, 2, 3, 4, 1]

    @testset "eta = 0 reproduces the plain statistics" begin
        ss0 = suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw), standardize = false,
            error_eta = 0.0, edge_index = eidx)
        ssp = suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw), standardize = false)
        @test ss0.K == ssp.K
        @test Matrix(ss0.design.VtV) ≈ Matrix(ssp.design.VtV) atol = 1e-12
        @test ss0.design.projected_y ≈ ssp.design.projected_y atol = 1e-12
        @test ss0.design.ydot ≈ ssp.design.ydot atol = 1e-12
        @test ss0.weights.log_weight_sum ≈ 0.0 atol = 1e-12
    end

    @testset "dense likelihood reference at a planted eta" begin
        eta = 0.4
        ss = suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw), standardize = false,
            error_eta = eta, edge_index = eidx)
        model = BipartiteNormalizedModel(ss.A_prior; rho_limit = 0.99)
        rho, sa, sz, se = 0.25, 0.8, 0.5, 0.4
        params = [atanh(rho / 0.99), log(sa), log(sz), log(se)]
        obs_stats = BipartiteGMRF.objective_stats(model, ss, params)
        exact_nll = BipartiteGMRF.nll_exact_value(model, ss, params, obs_stats)

        n = ss.N_firms + ss.N_workers
        Vi = Int[]; Vj = Int[]; Vv = Float64[]
        for k in 1:K_in
            push!(Vi, k); push!(Vj, f[k]); push!(Vv, 1.0)
            push!(Vi, k); push!(Vj, ss.N_firms + w[k]); push!(Vv, 1.0)
        end
        V = sparse(Vi, Vj, Vv, K_in, n)
        R = zeros(K_in, K_in)
        for i in 1:K_in, j in 1:K_in
            f[i] == f[j] && (R[i, j] = eta^abs(eidx[i] - eidx[j]))
        end
        Q = BipartiteGMRF.model_precision(model, rho, sa, sz)
        Sigma = Matrix(V * inv(Matrix(Q)) * transpose(V)) + se^2 .* R
        dense_nll = 0.5 * (logdet(Symmetric(Sigma)) + dot(y, Sigma \ y))
        @test exact_nll ≈ dense_nll atol = 1e-8 rtol = 1e-8
    end

    @testset "estimated eta; reported on result; dof counts it" begin
        result = fit_mle(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw),
            error_eta = :estimate, edge_index = eidx,
            solver = ExactCholesky(optim_iters = 80, polish = false))
        @test isfinite(result.nll)
        @test isfinite(result.eta)
        @test -1.0 < result.eta < 1.0
        @test result.metadata.error_eta == result.eta
        @test "eta" in coefnames(result)
        @test dof(result) == 5  # rho, sa, sz, se + eta
    end

    @testset "estimated eta evaluates eta = 0 (no BIG_NLL discontinuity)" begin
        ss_est = suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw), standardize = false,
            error_eta = :estimate, edge_index = eidx)
        ss_fix = suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw), standardize = false,
            error_eta = 0.0, edge_index = eidx)
        model = BipartiteNormalizedModel(ss_est.A_prior; rho_limit = 0.99)
        rho, sa, sz, se = 0.25, 0.8, 0.5, 0.4
        p_est = [atanh(rho / 0.99), log(sa), log(sz), log(se), 0.0]  # eta = tanh(0) = 0
        p_fix = [atanh(rho / 0.99), log(sa), log(sz), log(se)]
        obs_est = BipartiteGMRF.objective_stats(model, ss_est, p_est)
        obs_fix = BipartiteGMRF.objective_stats(model, ss_fix, p_fix)
        nll_est = BipartiteGMRF.nll_exact_value(model, ss_est, p_est, obs_est)
        nll_fix = BipartiteGMRF.nll_exact_value(model, ss_fix, p_fix, obs_fix)
        @test nll_est ≈ nll_fix atol = 1e-10
        @test nll_est < BipartiteGMRF.BIG_NLL
    end

    @testset "composes with X (profiled mean structure)" begin
        X = hcat(ones(length(y)), Float64.(w))
        result = fit_mle(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw),
            error_eta = :estimate, edge_index = eidx, X = X,
            solver = ExactCholesky(optim_iters = 60, polish = false))
        @test result.beta !== nothing
        @test length(result.beta) == 2
        @test all(isfinite, result.beta)
    end

    @testset "guardrails" begin
        # mutually exclusive with error_cov and error_groups
        @test_throws ArgumentError suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw),
            error_cov = sparse(1.0 * LinearAlgebra.I, K_in, K_in),
            error_eta = 0.3, edge_index = eidx)
        @test_throws ArgumentError suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw),
            error_groups = f, error_eta = 0.3, edge_index = eidx)
        # requires :raw
        @test_throws ArgumentError suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :edge),
            error_eta = 0.3, edge_index = eidx)
        # edge_index required
        @test_throws ArgumentError suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw), error_eta = 0.3)
        # edge_index without error_eta is rejected
        @test_throws ArgumentError suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw), edge_index = eidx)
        # eta outside (-1, 1)
        @test_throws ArgumentError suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw),
            error_eta = 1.5, edge_index = eidx)
        # bad eta symbol
        @test_throws ArgumentError suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw),
            error_eta = :bogus, edge_index = eidx)
        # length mismatch
        @test_throws ArgumentError suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw),
            error_eta = 0.3, edge_index = eidx[1:3])
        # duplicate edge index within a firm (firm 1 becomes [1, 2, 2])
        eidx_dup = copy(eidx); eidx_dup[3] = 2
        @test_throws ArgumentError suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw),
            error_eta = 0.3, edge_index = eidx_dup)
        # non-contiguous edge index within a firm (firm 1 becomes [1, 2, 4])
        eidx_gap = copy(eidx); eidx_gap[3] = 4
        @test_throws ArgumentError suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw),
            error_eta = 0.3, edge_index = eidx_gap)
        # ExactCholesky only
        ss = suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw),
            error_eta = 0.3, edge_index = eidx)
        @test_throws ArgumentError fit_mle(BipartiteNormalizedModel, ss;
            solver = HutchSLQ())
    end

    @testset "match-grouped spells (issue #120)" begin
        # ── one match per row reproduces the raw-row path exactly ──
        ss_raw = suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw), standardize = false,
            error_eta = 0.4, edge_index = eidx)
        ss_mid = suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw), standardize = false,
            error_eta = 0.4, edge_index = eidx, match_id = collect(1:K_in))
        @test ss_mid.K == ss_raw.K
        @test Matrix(ss_mid.design.VtV) ≈ Matrix(ss_raw.design.VtV) atol = 1e-14
        @test ss_mid.design.projected_y ≈ ss_raw.design.projected_y atol = 1e-14
        @test ss_mid.design.ydot ≈ ss_raw.design.ydot atol = 1e-14
        @test ss_mid.weights.log_weight_sum ≈ ss_raw.weights.log_weight_sum atol = 1e-14

        # ── dense likelihood reference on match units ──
        # Firm 1: match 1 co-managed (workers 1, 2; rank 1), match 2 (worker
        # 3; rank 2). Firm 2: match 3 (worker 2; rank 1), match 4 (worker 4;
        # rank 2). The error process lives on the 4 matches.
        fm = [1, 1, 1, 2, 2]
        wm = [1, 2, 3, 2, 4]
        ym = [0.7, 0.7, 0.9, 0.2, -0.5]
        midm = [1, 1, 2, 3, 4]
        eidxm = [1, 1, 2, 1, 2]
        eta = 0.4
        ssm = suffstats(BipartiteNormalizedModel, fm, wm, ym;
            weighting = Weighting(observations = :raw), standardize = false,
            error_eta = eta, edge_index = eidxm, match_id = midm)
        @test ssm.K == 4
        model = BipartiteNormalizedModel(ssm.A_prior; rho_limit = 0.99)
        rho, sa, sz, se = 0.25, 0.8, 0.5, 0.4
        params = [atanh(rho / 0.99), log(sa), log(sz), log(se)]
        obs_stats = BipartiteGMRF.objective_stats(model, ssm, params)
        exact_nll = BipartiteGMRF.nll_exact_value(model, ssm, params, obs_stats)

        n = ssm.N_firms + ssm.N_workers
        Vd = zeros(4, n)
        Vd[1, 1] = 1.0; Vd[1, ssm.N_firms + 1] = 0.5; Vd[1, ssm.N_firms + 2] = 0.5
        Vd[2, 1] = 1.0; Vd[2, ssm.N_firms + 3] = 1.0
        Vd[3, 2] = 1.0; Vd[3, ssm.N_firms + 2] = 1.0
        Vd[4, 2] = 1.0; Vd[4, ssm.N_firms + 4] = 1.0
        ymatch = [0.7, 0.9, 0.2, -0.5]
        firm_of = [1, 1, 2, 2]
        rank_of = [1, 2, 1, 2]
        R = [firm_of[s] == firm_of[t] ? eta^abs(rank_of[s] - rank_of[t]) : 0.0
             for s in 1:4, t in 1:4]
        Q = BipartiteGMRF.model_precision(model, rho, sa, sz)
        Sigma = Vd * inv(Matrix(Q)) * transpose(Vd) + se^2 .* R
        dense_nll = 0.5 * (logdet(Symmetric(Sigma)) + dot(ymatch, Sigma \ ymatch))
        @test exact_nll ≈ dense_nll atol = 1e-8 rtol = 1e-8

        # ── duplicating a member row of the co-managed match: fit unchanged ──
        result0 = fit_mle(BipartiteNormalizedModel, fm, wm, ym;
            weighting = Weighting(observations = :raw),
            error_eta = :estimate, edge_index = eidxm, match_id = midm,
            solver = ExactCholesky(optim_iters = 60, polish = false))
        resultd = fit_mle(BipartiteNormalizedModel,
            vcat(fm, 1), vcat(wm, 2), vcat(ym, 0.7);
            weighting = Weighting(observations = :raw),
            error_eta = :estimate, edge_index = vcat(eidxm, 1),
            match_id = vcat(midm, 1),
            solver = ExactCholesky(optim_iters = 60, polish = false))
        @test resultd.nll ≈ result0.nll atol = 1e-12
        @test resultd.rho ≈ result0.rho atol = 1e-12
        @test resultd.sigma_a ≈ result0.sigma_a atol = 1e-12
        @test resultd.sigma_z ≈ result0.sigma_z atol = 1e-12
        @test resultd.sigma_epsilon ≈ result0.sigma_epsilon atol = 1e-12
        @test resultd.eta ≈ result0.eta atol = 1e-12

        # ── composes with X on match units ──
        Xm = hcat(ones(length(ym)), Float64.(wm))
        resultx = fit_mle(BipartiteNormalizedModel, fm, wm, ym;
            weighting = Weighting(observations = :raw),
            error_eta = :estimate, edge_index = eidxm, match_id = midm,
            X = Xm, solver = ExactCholesky(optim_iters = 40, polish = false))
        @test resultx.beta !== nothing
        @test all(isfinite, resultx.beta)

        # ── guardrails: multi-firm match; edge_index disagreement in a match ──
        @test_throws ArgumentError suffstats(BipartiteNormalizedModel,
            [1, 2], [1, 1], [1.0, 1.0];
            weighting = Weighting(observations = :raw),
            error_eta = 0.3, edge_index = [1, 1], match_id = [1, 1])
        @test_throws ArgumentError suffstats(BipartiteNormalizedModel,
            [1, 1], [1, 2], [1.0, 1.0];
            weighting = Weighting(observations = :raw),
            error_eta = 0.3, edge_index = [1, 2], match_id = [1, 1])
    end
end