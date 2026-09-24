# Issue #122: symbolic support must not depend on which entries happen to
# cancel at the reference parameters. The likelihood oracle below works in
# observation space; it does not use the package's sparse cross-products.

function exact_pattern_dense_precision(model, rho, sa, sz)
    A = Matrix(model.graph.A)
    df = vec(sum(A; dims=2))
    dw = vec(sum(A; dims=1))
    if model isa BipartiteNormalizedModel
        W = A ./ sqrt.(df * dw')
        return [Matrix(Diagonal(fill(1 / sa^2, length(df)))) -rho .* W ./ (sa * sz);
                -rho .* W' ./ (sa * sz) Matrix(Diagonal(fill(1 / sz^2, length(dw))))]
    elseif model isa BipartiteUnnormalizedModel
        return [Matrix(Diagonal(df ./ sa^2)) -rho .* A ./ (sa * sz);
                -rho .* A' ./ (sa * sz) Matrix(Diagonal(dw ./ sz^2))]
    else
        @assert model isa BipartiteVarianceStableModel
        return [Matrix(Diagonal((1 .+ rho^2 .* (df .- 1)) ./ sa^2)) -rho .* A ./ (sa * sz);
                -rho .* A' ./ (sa * sz) Matrix(Diagonal((1 .+ rho^2 .* (dw .- 1)) ./ sz^2))] ./ (1 - rho^2)
    end
end

function exact_pattern_reference(model, V, y, firm, rank, rho, sa, sz, se, eta)
    Q = exact_pattern_dense_precision(model, rho, sa, sz)
    R = [firm[i] == firm[j] ? eta^abs(rank[i] - rank[j]) : 0.0
         for i in eachindex(y), j in eachindex(y)]
    P = V' * (R \ V)
    b = V' * (R \ y)
    M = Q + P / se^2
    Sigma = Symmetric(V * (Q \ V') + se^2 * R)
    nll = 0.5 * (logdet(Sigma) + dot(y, Sigma \ y))
    return (; Q, P, b, M, nll)
end

function exact_pattern_cache_state(cache)
    return map((cache.ws_Q, cache.ws_M)) do ws
        (; workspace=ws, matrix=ws.Q, columns=ws.Q.colptr,
           rows=ws.Q.rowval, values=nonzeros(ws.Q), backend=ws.backend,
           factor=ws.backend.factor,
           pattern=(copy(ws.Q.colptr), copy(ws.Q.rowval)))
    end
end

function test_exact_pattern_cache_state(cache, before)
    for (ws, original) in zip((cache.ws_Q, cache.ws_M), before)
        @test ws === original.workspace
        @test ws.Q === original.matrix
        @test ws.Q.colptr === original.columns
        @test ws.Q.rowval === original.rows
        @test nonzeros(ws.Q) === original.values
        @test ws.backend === original.backend
        @test ws.backend.factor === original.factor
        @test (ws.Q.colptr, ws.Q.rowval) == original.pattern
    end
end

function test_exact_pattern_evaluation(model, stats, cache, params, ref)
    bg = BipartiteGMRF
    gm = bg.GaussianMarkovRandomFields
    obs = bg.objective_stats(model, stats, params)
    value = bg.nll_exact_value(model, stats, params, obs, cache)
    fresh = bg.nll_exact_value(model, stats, params, obs)
    @test value < bg.BIG_NLL
    @test value ≈ ref.nll atol=1e-12 rtol=1e-12
    @test fresh ≈ ref.nll atol=1e-12 rtol=1e-12
    @test Matrix(obs.design.VtV) ≈ ref.P atol=1e-12 rtol=1e-12
    @test obs.design.projected_y ≈ ref.b atol=1e-12 rtol=1e-12
    @test Matrix(cache.ws_Q.Q) ≈ ref.Q atol=1e-12 rtol=1e-12
    @test Matrix(cache.ws_M.Q) ≈ ref.M atol=1e-12 rtol=1e-12
    # Fresh sparse factorizations independently check the cached determinants
    # and solve, in addition to the observation-space likelihood above.
    q_factor = cholesky(Symmetric(sparse(ref.Q)))
    m_factor = cholesky(Symmetric(sparse(ref.M)))
    @test -gm.logdet_cov(cache.ws_Q) ≈ logdet(q_factor) atol=1e-11 rtol=1e-11
    @test -gm.logdet_cov(cache.ws_M) ≈ logdet(m_factor) atol=1e-11 rtol=1e-11
    x = gm.workspace_solve(cache.ws_M, obs.design.projected_y)
    @test x ≈ m_factor \ ref.b atol=1e-11 rtol=1e-11
    @test dot(ref.b, x) ≈ dot(ref.b, ref.M \ ref.b) atol=1e-11 rtol=1e-11
    return nothing
end

@testset "ExactCholesky structural support (issue #122)" begin
    bg = BipartiteGMRF
    priors = (BipartiteNormalizedModel, BipartiteUnnormalizedModel,
              BipartiteVarianceStableModel)
    # The first match contains workers 1 and 2; the second contains worker 1.
    # At eta=0.5, the worker-worker cross-product is exactly zero, but becomes
    # nonzero when eta changes. This is the minimal silent-truncation fixture.
    f, w, y = [1, 1, 1], [1, 2, 1], [0.3, 0.3, 1.2]
    mid, rank = [1, 1, 2], [1, 1, 2]
    V = [1.0 0.5 0.5; 1.0 1.0 0.0]
    yobs = [0.3, 1.2]
    firmobs, rankobs = [1, 1], [1, 2]
    cases = [(0.2, 0.8, 0.6, 0.5, eta) for eta in (0.0, 0.5, 0.51, -0.3)]
    append!(cases, [(0.0, 0.8, 0.6, 0.5, 0.51),
                    (0.35, 1.2, 0.4, 0.8, 0.7),
                    (-0.2, 0.7, 1.1, 0.3, -0.4)])

    for prior in priors
        @testset "$prior: overlapping matches" begin
            stats_for(eta) = suffstats(prior, f, w, y;
                n_firms=1, n_workers=2, standardize=false, match_id=mid,
                edge_index=rank, error_eta=eta, weighting=Weighting(observations=:raw))
            ss = stats_for(:estimate)
            model = prior(ss.A_prior; rho_limit=0.8)
            @test ss.design.VtV[2, 3] == 0.0
            @test ss.error_ar1.pattern[2, 3] != 0.0

            for (order, sequence) in ((:forward, cases), (:reverse, reverse(cases)))
                @testset "estimated eta, $order cache reuse" begin
                    cache = bg.make_exact_workspace(model, ss)
                    before = exact_pattern_cache_state(cache)
                    # No worker-worker prior edge exists, but the posterior
                    # workspace must reserve it despite its zero seed value.
                    @test nnz(cache.ws_Q.Q) == 7
                    @test nnz(cache.ws_M.Q) == 9
                    for (rho, sa, sz, se, eta) in sequence
                        params = [atanh(rho / 0.8), log(sa), log(sz), log(se), atanh(eta)]
                        ref = exact_pattern_reference(model, V, yobs, firmobs, rankobs,
                                                      rho, sa, sz, se, eta)
                        test_exact_pattern_evaluation(model, ss, cache, params, ref)
                        test_exact_pattern_cache_state(cache, before)
                    end
                end
            end

            @testset "fixed eta, including the cancellation point" begin
                for eta in (0.0, 0.5, 0.51, -0.3)
                    fixed = stats_for(eta)
                    cache = bg.make_exact_workspace(model, fixed)
                    before = exact_pattern_cache_state(cache)
                    for (rho, sa, sz, se) in ((0.2, 0.8, 0.6, 0.5),
                                              (0.0, 1.2, 0.4, 0.8),
                                              (-0.2, 0.7, 1.1, 0.3))
                        params = [atanh(rho / 0.8), log(sa), log(sz), log(se)]
                        ref = exact_pattern_reference(model, V, yobs, firmobs, rankobs,
                                                      rho, sa, sz, se, eta)
                        test_exact_pattern_evaluation(model, fixed, cache, params, ref)
                        test_exact_pattern_cache_state(cache, before)
                    end
                end
            end

            @testset "ordinary independent errors, including rho zero" begin
                iid = suffstats(prior, f, w, y; standardize=false, match_id=mid,
                                weighting=Weighting(observations=:raw))
                cache = bg.make_exact_workspace(model, iid)
                before = exact_pattern_cache_state(cache)
                for rho in (0.2, 0.0, -0.2, 0.2)
                    params = [atanh(rho / 0.8), log(0.8), log(0.6), log(0.5)]
                    ref = exact_pattern_reference(model, V, yobs, firmobs, rankobs,
                                                  rho, 0.8, 0.6, 0.5, 0.0)
                    test_exact_pattern_evaluation(model, iid, cache, params, ref)
                    test_exact_pattern_cache_state(cache, before)
                end
            end
        end
    end

    @testset "raw-row AR(1), singleton firms, and graph-only edges" begin
        # Firm 2 has a singleton outcome; worker 4 only appears in the prior.
        # All three supported exact priors see an acyclic graph.
        fr, wr = [1, 1, 2, 1], [1, 2, 3, 4]
        yr, ranks = [0.2, -0.4, 0.8, NaN], [1, 2, 1, 3]
        Vr = [1.0 0.0 1.0 0.0 0.0 0.0;
              1.0 0.0 0.0 1.0 0.0 0.0;
              0.0 1.0 0.0 0.0 1.0 0.0]
        for prior in priors
            ss = suffstats(prior, fr, wr, yr; standardize=false,
                error_eta=:estimate, edge_index=ranks,
                weighting=Weighting(observations=:raw))
            model = prior(ss.A_prior; rho_limit=0.8)
            @test ss.metadata.graph_only_edges == 1
            @test ss.K == 3
            cache = bg.make_exact_workspace(model, ss)
            before = exact_pattern_cache_state(cache)
            for (rho, eta) in ((0.2, 0.5), (0.0, 0.0), (-0.2, -0.4), (0.2, 0.51))
                params = [atanh(rho / 0.8), log(0.8), log(0.6), log(0.5), atanh(eta)]
                ref = exact_pattern_reference(model, Vr, yr[1:3], fr[1:3], ranks[1:3],
                                              rho, 0.8, 0.6, 0.5, eta)
                test_exact_pattern_evaluation(model, ss, cache, params, ref)
                test_exact_pattern_cache_state(cache, before)
            end
        end
    end

    @testset "prior and observation contributions may cancel" begin
        ss = suffstats(BipartiteNormalizedModel, [1], [1], [0.7]; standardize=false)
        model = BipartiteNormalizedModel(ss.A_prior; rho_limit=0.8)
        cache = bg.make_exact_workspace(model, ss)
        before = exact_pattern_cache_state(cache)
        for rho in (0.2, 0.0, -0.2, 0.2)
            # lambda=0.25 and rho=0.25 cancel exactly when both prior scales
            # are one; also visit neighboring values with the same cache.
            for r in (rho, 0.25)
                params = [atanh(r / 0.8), 0.0, 0.0, log(2.0)]
                ref = exact_pattern_reference(model, [1.0 1.0], [0.7], [1], [1],
                                              r, 1.0, 1.0, 2.0, 0.0)
                test_exact_pattern_evaluation(model, ss, cache, params, ref)
                test_exact_pattern_cache_state(cache, before)
            end
        end
    end

    @testset "structural union covers prior-data cancellation (issue #107)" begin
        # At the old rho_ref=0.1 the unnormalized prior contributes -0.1 to each
        # firm-worker entry; this ten-worker match contributes exactly +0.1.
        # The reference numerical sum loses those positions, but the workspace
        # must retain them for all subsequent parameter evaluations.
        ss = suffstats(BipartiteUnnormalizedModel, ones(Int, 10), collect(1:10), fill(0.7, 10);
            match_id=ones(Int, 10), standardize=false)
        model = BipartiteUnnormalizedModel(ss.A_prior; rho_limit=0.8)
        Q0 = bg.model_precision(model, 0.1, 1.0, 1.0)
        M0 = Q0 + ss.design.VtV
        @test M0[1, 2] == 0.0
        cache = bg.make_exact_workspace(model, ss)
        @test nnz(cache.ws_M.Q) > nnz(M0)
        Qseed = bg.model_precision(model, 0.0, 1.0, 1.0)
        @test Matrix(cache.ws_Q.Q) == Matrix(Qseed)
        @test Matrix(cache.ws_M.Q) == Matrix(Qseed + ss.design.VtV)
        before = exact_pattern_cache_state(cache)
        Vmatch = hcat(ones(1, 1), fill(0.1, 1, 10))
        for rho in (0.2, 0.0, 0.1, -0.2)
            params = [atanh(rho / 0.8), 0.0, 0.0, 0.0]
            ref = exact_pattern_reference(model, Vmatch, [0.7], [1], [1],
                                          rho, 1.0, 1.0, 1.0, 0.0)
            test_exact_pattern_evaluation(model, ss, cache, params, ref)
            test_exact_pattern_cache_state(cache, before)
        end
    end

    @testset "fitted estimated-eta likelihood agrees with an independent oracle" begin
        rng = MersenneTwister(20260924)
        nf, nw, spells = 24, 48, 4
        ff, ww, matches, ranks = Int[], Int[], Int[], Int[]
        K = nf * spells
        Vf = zeros(K, nf + nw)
        # Each firm's first two matches share worker a. Worker b appears
        # nowhere else, preserving the eta=0.5 cancellation. Later matches
        # connect firms and make this a nontrivial estimation problem.
        for i in 1:nf
            a, b = mod1(i, 12), 12 + i
            c, d = 36 + mod1(i + 3, 12), mod1(i + 5, 12)
            for (r, members) in enumerate(([a, b], [a], [c], [d]))
                k = (i - 1) * spells + r
                Vf[k, i] = 1.0
                for worker in members
                    Vf[k, nf + worker] = inv(length(members))
                    push!(ff, i); push!(ww, worker)
                    push!(matches, k); push!(ranks, r)
                end
            end
        end
        seedstats = suffstats(BipartiteNormalizedModel, ff, ww, zeros(length(ff));
            n_firms=nf, n_workers=nw, standardize=false, match_id=matches,
            edge_index=ranks, error_eta=:estimate, weighting=Weighting(observations=:raw))
        model = BipartiteNormalizedModel(seedstats.A_prior; rho_limit=0.9)
        rho, sa, sz, se, eta = 0.25, 0.8, 0.6, 0.5, 0.35
        Q = exact_pattern_dense_precision(model, rho, sa, sz)
        firms = repeat(1:nf; inner=spells)
        order = repeat(1:spells; outer=nf)
        R = [firms[i] == firms[j] ? eta^abs(order[i] - order[j]) : 0.0
             for i in 1:K, j in 1:K]
        Sigma = Symmetric(Vf * (Q \ Vf') + se^2 * R)
        outcomes = cholesky(Sigma).L * randn(rng, K)
        result = fit_mle(BipartiteNormalizedModel, ff, ww, outcomes[matches];
            n_firms=nf, n_workers=nw, standardize=false, match_id=matches,
            edge_index=ranks, error_eta=:estimate, weighting=Weighting(observations=:raw),
            rho_limit=0.9, init=(rho=rho, sigma_a=sa, sigma_z=sz, sigma_epsilon=se, eta=eta),
            solver=ExactCholesky(optim_iters=600, polish=true), seed=20260924)
        ref = exact_pattern_reference(result.model, Vf, outcomes, firms, order,
            result.rho, result.sigma_a, result.sigma_z, result.sigma_epsilon, result.eta)
        # Test the returned likelihood at the returned parameters, not a
        # particular optimum or convergence trajectory across platforms.
        @test isfinite(result.nll)
        @test result.nll < bg.BIG_NLL
        @test isfinite(result.eta)
        @test result.nll ≈ ref.nll atol=1e-7 rtol=1e-9
    end
end
