# Warm starts (init=, issue #114). `init` replaces the hardcoded heuristic in
# initial_params field by field, in ORIGINAL outcome units. Two things are
# worth testing sharply and separately:
#
#   1. The starting point itself. It is not observable through fit_mle —
#      `theta_unconstrained` on the result is the *minimizer*, and Nelder-Mead
#      builds its full simplex even at optim_iters = 1 — so the assertions go
#      through `initial_point`, which is exactly the vector optimize_problem
#      hands to the optimizer.
#   2. That init = nothing did not disturb the default path. The goldens below
#      are the pre-#114 literal vectors, asserted with == rather than ≈.


@testset "warm starts (init)" begin
    d = synthetic_data()
    const_solver = ExactCholesky(optim_iters = 40, polish = false)

    @testset "defaults unchanged when init is absent" begin
        # The pre-#114 construction, verbatim. default_rho_start(L) =
        # min(0.5, 0.5L), so the rho code is atanh(0.5) for every L <= 1.
        @test BipartiteGMRF.initial_params(nothing, false; rho_limit = 0.99) ==
            [atanh(0.5), log(0.7), log(0.04), log(0.4)]
        @test BipartiteGMRF.initial_params(nothing, false; rho_limit = 0.5) ==
            [atanh(0.5), log(0.7), log(0.04), log(0.4)]
        @test BipartiteGMRF.initial_params(0.3, false; rho_limit = 0.99) ==
            [log(0.7), log(0.04), log(0.4)]
        @test BipartiteGMRF.initial_params(nothing, true; rho_limit = 0.99) ==
            [atanh(0.5), log(0.7), log(0.04), log(0.4),
             BipartiteGMRF.rhoeps_to_unconstrained(0.5)]
        # ... and passing init = nothing explicitly changes nothing.
        @test BipartiteGMRF.initial_params(nothing, false; rho_limit = 0.99, init = nothing) ==
            BipartiteGMRF.initial_params(nothing, false; rho_limit = 0.99)

        ss = suffstats(BipartiteNormalizedModel, d.f, d.w, d.y)
        @test BipartiteGMRF.initial_point(ss, nothing, false, 0, false, nothing;
            rho_limit = 0.99) == [atanh(0.5), log(0.7), log(0.04), log(0.4)]

        # The plumbing itself is inert: adding the kwarg must not perturb a fit.
        r0 = fit_mle(BipartiteNormalizedModel, d.f, d.w, d.y; solver = const_solver, seed = 1)
        r1 = fit_mle(BipartiteNormalizedModel, d.f, d.w, d.y; solver = const_solver, seed = 1,
            init = nothing)
        @test r0.nll == r1.nll
        @test r0.theta_unconstrained == r1.theta_unconstrained
    end

    @testset "the optimizer starts at init, in original outcome units" begin
        ss = suffstats(BipartiteNormalizedModel, d.f, d.w, d.y)   # standardize = true
        @test ss.y_std != 1.0                                     # the test has teeth
        init = (rho = 0.55, sigma_a = 1.4, sigma_z = 0.3, sigma_epsilon = 1.1)
        p0 = BipartiteGMRF.initial_point(ss, nothing, false, 0, false, init; rho_limit = 0.99)
        dec = BipartiteGMRF.unpack_params(p0; rho_limit = 0.99)
        @test dec.rho ≈ 0.55
        @test dec.sigma_a * ss.y_std ≈ 1.4
        @test dec.sigma_z * ss.y_std ≈ 0.3
        @test dec.sigma_epsilon * ss.y_std ≈ 1.1

        # standardize = false: the two unit conventions coincide.
        ssr = suffstats(BipartiteNormalizedModel, d.f, d.w, d.y;
            weighting = Weighting(observations = :raw), standardize = false)
        @test ssr.y_std == 1.0
        q0 = BipartiteGMRF.initial_point(ssr, nothing, false, 0, false, init; rho_limit = 0.99)
        @test BipartiteGMRF.unpack_params(q0; rho_limit = 0.99).sigma_a ≈ 1.4
    end

    @testset "partial init keeps the defaults for the rest" begin
        ss = suffstats(BipartiteNormalizedModel, d.f, d.w, d.y)
        p0 = BipartiteGMRF.initial_point(ss, nothing, false, 0, false, (rho = 0.7,);
            rho_limit = 0.99)
        @test p0[1] == atanh(0.7 / 0.99)
        @test p0[2:4] == [log(0.7), log(0.04), log(0.4)]

        # A nothing-valued field counts as not supplied, which is what makes
        # init = params(result) work without hand-editing.
        p1 = BipartiteGMRF.initial_point(ss, nothing, false, 0, false,
            (rho = nothing, sigma_a = nothing, rho_eps = nothing, eta = nothing,
             beta = nothing); rho_limit = 0.99)
        @test p1 == BipartiteGMRF.initial_point(ss, nothing, false, 0, false, nothing;
            rho_limit = 0.99)
    end

    @testset "round trip through params(result)" begin
        r0 = fit_mle(BipartiteNormalizedModel, d.f, d.w, d.y; solver = const_solver, seed = 1)
        r1 = fit_mle(BipartiteNormalizedModel, d.f, d.w, d.y; solver = const_solver, seed = 1,
            init = params(r0))
        @test isfinite(r1.nll)
        # Restarting at the previous optimum cannot make the objective worse:
        # p0 is now the old minimizer, and Nelder-Mead returns its best vertex.
        @test r1.nll <= r0.nll + 1e-8
        @test r1.obj_evals <= r0.obj_evals

        # The starting point really is the previous fit's answer.
        ss = suffstats(BipartiteNormalizedModel, d.f, d.w, d.y)
        p0 = BipartiteGMRF.initial_point(ss, nothing, false, 0, false, params(r0);
            rho_limit = 0.99)
        @test p0 ≈ r0.theta_unconstrained
    end

    @testset "validation" begin
        fit(init) = fit_mle(BipartiteNormalizedModel, d.f, d.w, d.y;
            solver = const_solver, seed = 1, init = init)

        @test_throws ArgumentError fit((sigma_eps = 1.0,))          # typo
        @test_throws ArgumentError fit((rho = 0.995,))              # |rho| >= rho_limit
        @test_throws ArgumentError fit((rho = -1.5,))
        @test_throws ArgumentError fit((sigma_a = 0.0,))            # the degenerate optimum
        @test_throws ArgumentError fit((sigma_z = -1.0,))
        @test_throws ArgumentError fit((sigma_epsilon = Inf,))
        @test_throws ArgumentError fit((sigma_a = "0.7",))          # not a number
        @test_throws ArgumentError fit((eta = 0.3,))                # block not in this fit
        @test_throws ArgumentError fit((omega = [2.0],))
        @test_throws ArgumentError fit((phi = 0.5,))
        @test_throws ArgumentError fit((r = 0.2,))
        @test_throws ArgumentError fit((delta = 8.0,))

        # rho_eps clamps silently at 0.999 in the codec, so it is rejected
        # rather than quietly relocated.
        ss_eff = suffstats(BipartiteNormalizedModel, d.f, d.w, d.y;
            weighting = Weighting(observations = :effective, rho_eps = :estimate))
        @test_throws ArgumentError BipartiteGMRF.initial_point(ss_eff, nothing, true, 0, false,
            (rho_eps = 0.9995,); rho_limit = 0.99)
        p_eff = BipartiteGMRF.initial_point(ss_eff, nothing, true, 0, false,
            (rho_eps = 0.3,); rho_limit = 0.99)
        @test length(p_eff) == 5
        @test BipartiteGMRF.rhoeps_from_unconstrained(p_eff[5]) ≈ 0.3

        # A numerically fixed rho_eps is present but pinned: warn and ignore.
        ss_fix = suffstats(BipartiteNormalizedModel, d.f, d.w, d.y;
            weighting = Weighting(observations = :effective, rho_eps = 0.3))
        @test_logs (:warn, r"init.rho_eps is ignored") BipartiteGMRF.initial_point(
            ss_fix, nothing, false, 0, false, (rho_eps = 0.7,); rho_limit = 0.99)
        # ... but not when it merely repeats the value already in effect, which
        # is what init = params(result) does on an identically configured fit.
        @test_logs min_level = Base.CoreLogging.Warn BipartiteGMRF.initial_point(
            ss_fix, nothing, false, 0, false, (rho_eps = 0.3,); rho_limit = 0.99)

        # A status that forgets a key must fail loud, never silently drop the
        # field: that is the failure this validation exists to prevent.
        @test_throws ArgumentError BipartiteGMRF.validate_init((phi = 0.5,), (rho = :free,);
            rho_limit = 0.99)
    end

    @testset "fix_rho wins over init.rho, with a warning" begin
        ss = suffstats(BipartiteNormalizedModel, d.f, d.w, d.y)
        p0 = @test_logs (:warn, r"init.rho is ignored") BipartiteGMRF.initial_point(
            ss, 0.2, false, 0, false, (rho = 0.7, sigma_a = 1.4); rho_limit = 0.99)
        @test length(p0) == 3                       # no rho slot under fix_rho
        @test p0[1] ≈ log(1.4 / ss.y_std)           # the rest of init still applies
        # An out-of-range rho is not even checked when it is being ignored.
        @test_logs (:warn, r"init.rho is ignored") BipartiteGMRF.initial_point(
            ss, 0.2, false, 0, false, (rho = 5.0,); rho_limit = 0.99)
    end

    @testset "error_groups: the omega ladder" begin
        f = [1, 2, 2, 3, 3, 3, 4]
        w = [1, 2, 3, 1, 2, 4, 5]
        y = [0.5, -0.2, 0.9, 0.1, -0.7, 0.4, 1.2]
        ss = suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw), standardize = false,
            error_groups = f)
        n_omega = length(ss.error_classes.counts) - 1
        @test n_omega == 2

        # Free classes 2..C, and the full ladder as reported in
        # error_class_variances (leading 1.0), are both accepted.
        p_free = BipartiteGMRF.initial_point(ss, nothing, false, n_omega, false,
            (omega = [2.0, 4.0],); rho_limit = 0.99)
        p_ladder = BipartiteGMRF.initial_point(ss, nothing, false, n_omega, false,
            (omega = [1.0, 2.0, 4.0],); rho_limit = 0.99)
        @test p_free == p_ladder
        @test exp.(p_free[5:6]) ≈ [2.0, 4.0]
        # Default is omega = 1 for every class.
        @test BipartiteGMRF.initial_point(ss, nothing, false, n_omega, false, nothing;
            rho_limit = 0.99)[5:6] == [0.0, 0.0]

        @test_throws ArgumentError BipartiteGMRF.init_omega_codes((omega = [2.0, 4.0, 8.0, 16.0],), n_omega)
        @test_throws ArgumentError BipartiteGMRF.init_omega_codes((omega = [2.0, 0.0],), n_omega)
        @test_throws ArgumentError BipartiteGMRF.init_omega_codes((omega = [1.5, 2.0, 4.0],), n_omega)
        @test_throws ArgumentError BipartiteGMRF.init_omega_codes((omega = 2.0,), n_omega)

        res = fit_mle(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw), standardize = false,
            error_groups = f, solver = ExactCholesky(optim_iters = 20, polish = false),
            init = (rho = 0.3, sigma_a = 0.8, omega = [1.5, 2.0]))
        @test isfinite(res.nll)
        @test length(res.metadata.error_class_variances) == n_omega + 1
    end

    @testset "error_eta" begin
        f = [1, 1, 1, 2, 2, 3, 4, 4, 4, 4, 5]
        w = [1, 2, 3, 2, 4, 5, 1, 6, 7, 3, 7]
        y = [0.7, -0.3, 0.9, 0.2, -0.5, 1.1, -0.8, 0.4, 0.4, -0.2, 0.6]
        eidx = [1, 2, 3, 1, 2, 1, 1, 2, 3, 4, 1]
        ss = suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw), standardize = false,
            error_eta = :estimate, edge_index = eidx)

        p0 = BipartiteGMRF.initial_point(ss, nothing, false, 0, true, (eta = -0.3,);
            rho_limit = 0.99)
        @test length(p0) == 5
        @test BipartiteGMRF.eta_from_unconstrained(p0[5]) ≈ -0.3
        @test BipartiteGMRF.initial_point(ss, nothing, false, 0, true, nothing;
            rho_limit = 0.99)[5] == BipartiteGMRF.eta_to_unconstrained(0.5)
        # |eta| >= 1 is caught by the codec, which already has the message.
        @test_throws ArgumentError BipartiteGMRF.initial_point(ss, nothing, false, 0, true,
            (eta = 1.0,); rho_limit = 0.99)

        res = fit_mle(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw), standardize = false,
            error_eta = :estimate, edge_index = eidx,
            solver = ExactCholesky(optim_iters = 20, polish = false),
            init = (rho = 0.2, eta = 0.1))
        @test isfinite(res.nll)
        @test -1 < res.eta < 1

        # A numerically fixed eta is present but pinned: warn and ignore.
        ss_fixed = suffstats(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw), standardize = false,
            error_eta = 0.4, edge_index = eidx)
        @test_logs (:warn, r"init.eta is ignored") BipartiteGMRF.initial_point(
            ss_fixed, nothing, false, 0, false, (eta = 0.1,); rho_limit = 0.99)
        # Re-fitting the same configuration with init = params(result) carries
        # the pinned eta along and must not warn about it.
        r_fixed = fit_mle(BipartiteNormalizedModel, f, w, y;
            weighting = Weighting(observations = :raw), standardize = false,
            error_eta = 0.4, edge_index = eidx,
            solver = ExactCholesky(optim_iters = 10, polish = false))
        @test r_fixed.eta == 0.4
        @test_logs min_level = Base.CoreLogging.Warn BipartiteGMRF.initial_point(
            ss_fixed, nothing, false, 0, false, params(r_fixed); rho_limit = 0.99)
    end

    @testset "error_blocks = :iw (EMIWBlocks)" begin
        fb = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
        wb = [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 1]
        rng = MersenneTwister(7)
        yb = randn(rng, length(fb))
        ss = suffstats(BipartiteVarianceStableModel, fb, wb, yb;
            weighting = Weighting(observations = :raw), standardize = false,
            error_blocks = :iw, firm_group = fb)
        model = BipartiteVarianceStableModel(ss.A_prior; rho_limit = 0.99)
        solver = EMIWBlocks(max_iter = 30, ftol = 1e-8)

        fit = BipartiteGMRF.optimize_emiw(model, ss, solver;
            init = (rho = 0.3, sigma_a = 0.6, sigma_z = 0.2, phi = 0.4))
        @test isfinite(fit.nll)
        @test fit.phi > 0

        # sigma_epsilon is not an EMIW coordinate; it converts to the scale phi
        # through omega_bar = phi*delta/(delta-2).
        fit2 = BipartiteGMRF.optimize_emiw(model, ss, EMIWBlocks(max_iter = 1, delta = 10.0);
            init = (sigma_epsilon = 0.5, r = 0.0))
        @test isfinite(fit2.nll)

        # phi wins over sigma_epsilon, and says so rather than dropping it.
        @test_logs (:warn, r"init.sigma_epsilon is ignored") match_mode = :any BipartiteGMRF.optimize_emiw(
            model, ss, EMIWBlocks(max_iter = 1); init = (phi = 0.4, sigma_epsilon = 0.5))

        # A solver-fixed r or delta is pinned: warn and ignore, unless the value
        # supplied is the one already in effect.
        @test_logs (:warn, r"init.r is ignored") match_mode = :any BipartiteGMRF.optimize_emiw(
            model, ss, EMIWBlocks(max_iter = 1, r = 0.0); init = (r = 0.3,))
        @test_logs min_level = Base.CoreLogging.Warn match_mode = :any BipartiteGMRF.validate_init(
            (r = 0.0, delta = 12.0),
            BipartiteGMRF.emiw_init_status(EMIWBlocks(r = 0.0, delta = 12.0), 3);
            rho_limit = 0.99)

        # Blocks that belong to other error models are absent here.
        @test_throws ArgumentError BipartiteGMRF.optimize_emiw(model, ss, solver;
            init = (eta = 0.2,))
        @test_throws ArgumentError BipartiteGMRF.optimize_emiw(model, ss, solver;
            init = (delta = 1.5,))

        # r is not estimated at all when every block is a singleton, so init.r
        # would become permanent rather than a start. Its message says why.
        fs = [1, 2, 3, 4, 5, 6]
        ws = [1, 2, 3, 4, 5, 6]
        ys = randn(MersenneTwister(3), 6)
        ss1 = suffstats(BipartiteVarianceStableModel, fs, ws, ys;
            weighting = Weighting(observations = :raw), standardize = false,
            error_blocks = :iw, firm_group = fs)
        model1 = BipartiteVarianceStableModel(ss1.A_prior; rho_limit = 0.99)
        @test maximum(ss1.error_blocks.sizes) == 1
        err = try
            BipartiteGMRF.optimize_emiw(model1, ss1, EMIWBlocks(max_iter = 1);
                init = (r = 0.2,))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("single observation", sprint(showerror, err))
    end
end
