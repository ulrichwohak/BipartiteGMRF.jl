# Group-robust errors: observations sharing a group id (typically the firm)
# are collapsed to their mean, and the group-mean error variance is
# sigma_eps^2 * omega_c with one free omega per group-size class, estimated
# inside the MLE (smallest class pinned at omega = 1). This makes the fit
# robust to an ARBITRARY unknown PD error covariance within groups — the
# unknown block enters the likelihood only through the group-mean variance,
# which omega absorbs. The sharpest test is a dense reference: the
# exact-solver NLL at planted omegas must equal the closed-form Gaussian NLL
# with Sigma = Vg Q^-1 Vg' + sigma^2 diag(omega_class), exactly.

@testset "group-robust errors (error_groups)" begin
    f = [1, 1, 1, 2, 2, 3, 4, 4, 4, 4, 5]
    w = [1, 2, 3, 2, 4, 5, 1, 6, 7, 3, 7]
    mid = [1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 10]
    y = [0.7, -0.3, 0.9, 0.2, -0.5, 1.1, -0.8, 0.4, 0.4, -0.2, 0.6]

    @testset "all-singleton groups reduce to the plain statistics" begin
        ss0 = suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw), standardize = false,
            error_groups = collect(1:length(y)))
        ssp = suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw), standardize = false)
        @test ss0.K == ssp.K
        @test length(ss0.error_classes.counts) == 1
        @test Matrix(ss0.design.VtV) ≈ Matrix(ssp.design.VtV) atol = 1e-12
        @test ss0.design.projected_y ≈ ssp.design.projected_y atol = 1e-12
        @test ss0.design.ydot ≈ ssp.design.ydot atol = 1e-12
    end

    @testset "dense likelihood reference at planted omegas" begin
        fd = [1, 2, 2, 3, 3, 3, 4]
        wd = [1, 2, 3, 1, 2, 4, 5]
        yd = [0.5, -0.2, 0.9, 0.1, -0.7, 0.4, 1.2]
        ss = suffstats(BipartiteNormalizedModel, fd, wd, yd;
            weighting = Weighting(observations = :raw), standardize = false,
            error_groups = fd)
        @test ss.K == 4                        # four firms -> four group means
        @test ss.error_classes.sizes == [1, 2, 3]
        @test ss.error_classes.counts == [2, 1, 1]

        model = BipartiteNormalizedModel(ss.A_prior; rho_limit = 0.99)
        rho, sa, sz, se = 0.25, 0.8, 0.5, 0.4
        om2, om3 = 1.8, 2.5                    # class 1 (singletons) pinned at 1
        params = [atanh(rho / 0.99), log(sa), log(sz), log(se), log(om2), log(om3)]
        obs_stats = BipartiteGMRF.objective_stats(model, ss, params)
        exact_nll = BipartiteGMRF.nll_exact_value(model, ss, params, obs_stats)

        n = ss.N_firms + ss.N_workers
        nf = ss.N_firms
        Vg = zeros(4, n)
        Vg[1, 1] = 1.0; Vg[1, nf + 1] = 1.0                        # firm 1: row 1
        Vg[2, 2] = 1.0; Vg[2, nf + 2] = 0.5; Vg[2, nf + 3] = 0.5   # firm 2: rows 2-3
        Vg[3, 3] = 1.0                                             # firm 3: rows 4-6
        Vg[3, nf + 1] = 1/3; Vg[3, nf + 2] = 1/3; Vg[3, nf + 4] = 1/3
        Vg[4, 4] = 1.0; Vg[4, nf + 5] = 1.0                        # firm 4: row 7
        yg = [yd[1], (yd[2] + yd[3]) / 2, (yd[4] + yd[5] + yd[6]) / 3, yd[7]]
        omega_of_group = [1.0, om2, om3, 1.0]

        Q = BipartiteGMRF.model_precision(model, rho, sa, sz)
        Sigma = Vg * inv(Matrix(Q)) * transpose(Vg) +
                se^2 .* Matrix(Diagonal(omega_of_group))
        dense_nll = 0.5 * (logdet(Symmetric(Sigma)) + dot(yg, Sigma \ yg))
        @test exact_nll ≈ dense_nll atol = 1e-8 rtol = 1e-8
    end

    @testset "fit_mle end to end; omegas reported; dof counts them" begin
        result = fit_mle(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw),
            error_groups = f,
            solver = ExactCholesky(optim_iters = 60, polish = false))
        @test isfinite(result.nll)
        @test isfinite(result.rho)
        # firm sizes 3,2,1,4,1 -> classes [1,2,3,4], three free omegas
        @test result.metadata.error_class_sizes == [1, 2, 3, 4]
        omhat = result.metadata.error_class_variances
        @test length(omhat) == 4
        @test omhat[1] == 1.0
        @test all(o -> isfinite(o) && o > 0, omhat)
        @test dof(result) == 4 + 3
    end

    @testset "composes with match_id" begin
        ss = suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw), match_id = mid,
            standardize = false, error_groups = f)
        # 10 matches at 5 firms: group sizes 3,2,1,3,1 (firm 4's four rows
        # are three matches — rows 8-9 are one co-CEO match)
        @test ss.K == 5
        @test ss.error_classes.sizes == [1, 2, 3]
        @test ss.error_classes.counts == [2, 1, 2]
        result = fit_mle(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw), match_id = mid,
            error_groups = f,
            solver = ExactCholesky(optim_iters = 40, polish = false))
        @test isfinite(result.nll)
    end

    @testset "composes with X (profiled mean structure)" begin
        X = hcat(ones(length(y)), Float64.(w))
        result = fit_mle(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw),
            error_groups = f, X = X,
            solver = ExactCholesky(optim_iters = 40, polish = false))
        @test result.beta !== nothing
        @test length(result.beta) == 2
        @test all(isfinite, result.beta)
    end

    @testset "guardrails" begin
        # mutually exclusive with error_cov
        @test_throws ArgumentError suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw),
            error_cov = sparse(1.0 * LinearAlgebra.I, length(y), length(y)),
            error_groups = f)
        # requires :raw
        @test_throws ArgumentError suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :edge), error_groups = f)
        # length mismatch
        @test_throws ArgumentError suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw), error_groups = f[1:3])
        # a match must not span two groups
        g_bad = copy(f); g_bad[9] = 99   # rows 8-9 share match 8
        @test_throws ArgumentError suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw), match_id = mid,
            error_groups = g_bad)
        # ExactCholesky only
        ss = suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw), error_groups = f)
        @test_throws ArgumentError fit_mle(BipartiteNormalizedModel, ss;
            solver = HutchSLQ())
    end
end
