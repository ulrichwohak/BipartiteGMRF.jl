@testset "inference" begin
    # Moderate, informative synthetic panel so the observed information is
    # well-conditioned. Behaviour is asserted structurally / via exact
    # delta-method identities, not by pinning one lucky seed.
    df, _ = simulate_gmrf_panel(101; n_firms=50, n_workers=50, n_edges=300, reps=4)
    fit = gmrf_mle(df; solver=ExactCholesky(optim_iters=150, polish=true),
        decompose=false, seed=1)
    @test isfinite(fit.nll)

    z = 1.959963984540054  # Phi^{-1}(0.975)

    @testset "stderror (ExactCholesky)" begin
        se = stderror(fit)
        @test se isa NamedTuple
        @test keys(se) == (:rho, :sigma_a, :sigma_z, :sigma_epsilon, :rho_eps)
        for nm in (:rho, :sigma_a, :sigma_z, :sigma_epsilon)
            @test se[nm] isa Float64
            @test isfinite(se[nm]) && se[nm] > 0
        end
        @test se.rho_eps === nothing  # rho_eps not estimated under default weighting
    end

    @testset "vcov and observed_information" begin
        V = vcov(fit)
        @test size(V) == (4, 4)
        @test isapprox(V, transpose(V); rtol=1e-8)
        @test all(>(0), diag(V))

        se = stderror(fit)
        @test isapprox(diag(V),
            [se.rho, se.sigma_a, se.sigma_z, se.sigma_epsilon] .^ 2; rtol=1e-6)

        H = observed_information(fit)
        @test size(H) == (4, 4)
        @test isapprox(H, transpose(H); rtol=1e-8)
    end

    @testset "confint respects constraints and matches stderror" begin
        se = stderror(fit)
        ci = confint(fit; level=0.95)
        limit = fit.problem.prior.rho_limit

        @test ci.rho[1] < fit.rho < ci.rho[2]
        @test -limit < ci.rho[1] && ci.rho[2] < limit
        for nm in (:sigma_a, :sigma_z, :sigma_epsilon)
            lo, hi = ci[nm]
            @test 0 < lo < getfield(fit, nm) < hi
            # Exact identity: log(hi/lo) == 2 z (se_natural / point) for log-scale params.
            @test isapprox(log(hi / lo), 2 * z * se[nm] / getfield(fit, nm); rtol=1e-6)
        end
        @test ci.rho_eps === nothing

        # Exact identity on the unconstrained scale: the atanh-width equals
        # 2 z se_theta, and se.rho = J_rho * se_theta with J_rho the tanh slope.
        j_rho = limit * (1 - (fit.rho / limit)^2)
        @test isapprox(atanh(ci.rho[2] / limit) - atanh(ci.rho[1] / limit),
            2 * z * se.rho / j_rho; rtol=1e-6)

        @test_throws ArgumentError confint(fit; level=1.5)
    end

    @testset "fixed rho drops the rho parameter" begin
        fixed = gmrf_mle(df; solver=ExactCholesky(optim_iters=120, polish=true),
            decompose=false, seed=1, fix_rho=0.3)
        sef = stderror(fixed)
        @test sef.rho === nothing
        @test isfinite(sef.sigma_a) && sef.sigma_a > 0
        @test size(vcov(fixed)) == (3, 3)
        @test size(observed_information(fixed)) == (3, 3)
        @test confint(fixed).rho === nothing
        @test occursin("(fixed)", sprint(show, MIME"text/plain"(), fixed))
    end

    @testset "estimated rho_eps extends the parameter vector" begin
        # The synthetic panel has i.i.d. residuals (true rho_eps = 0, a boundary),
        # so SEs for rho_eps are not well-defined here; assert only the parameter
        # bookkeeping (the rho_eps column/slot is present) and the SE shape.
        eff = gmrf_mle(df; solver=ExactCholesky(optim_iters=120, polish=true),
            weighting=Weighting(observations=:effective, rho_eps=:estimate),
            decompose=false, seed=1)
        @test size(vcov(eff)) == (5, 5)
        @test stderror(eff).rho_eps !== nothing
    end

    @testset "show includes SE annotations for ExactCholesky" begin
        @test occursin("±", sprint(show, MIME"text/plain"(), fit))
    end

    @testset "HutchSLQ requires an explicit opt-in" begin
        hutch = gmrf_mle(df;
            solver=HutchSLQ(logdet_probes=6, lanczos_iters=10, optim_iters=20, cg_maxiter=200),
            decompose=false, seed=1)
        @test_throws ArgumentError stderror(hutch)
        @test_throws ArgumentError vcov(hutch)
        local se_h
        @test_logs (:warn,) match_mode = :any (se_h = stderror(hutch; compute_se=true))
        @test se_h isa NamedTuple
        @test keys(se_h) == (:rho, :sigma_a, :sigma_z, :sigma_epsilon, :rho_eps)
        # HutchSLQ show stays cheap: no SEs computed, so no annotation.
        @test !occursin("±", sprint(show, MIME"text/plain"(), hutch))
    end
end
