# Independent observation-space references for the iterative AR(1) path.
# These helpers deliberately do not call the package's AR/mean-stat builders.
function hutch_ar1_fixture()
    f = [1, 1, 1, 1, 2, 2, 3, 3, 4, 4]
    w = [1, 2, 2, 3, 3, 4, 5, 6, 6, 7]
    y = [0.3, 0.3, 1.2, -0.2, 0.7, -0.5, 0.4, NaN, NaN, NaN]
    match = [10, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    rank = [1, 1, 2, 3, 1, 2, 1, 99, 99, 99]
    X = sparse(hcat(ones(length(y)), [0.1, 0.1, -0.7, 0.9, 0.2, 0.5, -0.2, 9, 8, 7]))
    return (; f, w, y, match, rank, X)
end

function hutch_ar1_dense_rows(data, X)
    observed = findall(isfinite, data.y)
    groups = [filter(i -> data.match[i] == m, observed)
              for m in unique(data.match[observed])]
    src = first.(groups)
    nf, nw = maximum(data.f), maximum(data.w)
    V = zeros(length(groups), nf + nw)
    for (j, rows) in enumerate(groups)
        firms, workers = unique(data.f[rows]), unique(data.w[rows])
        @assert length(firms) == 1
        V[j, firms] .= 1.0
        V[j, nf .+ workers] .= inv(length(workers))
    end
    # Binary prior support includes every row, including missing-outcome rows.
    A = zeros(nf, nw)
    for i in eachindex(data.f)
        A[data.f[i], data.w[i]] = 1.0
    end
    return (; V, A, y=data.y[src], X=X === nothing ? nothing : Matrix(X[src, :]),
            firms=data.f[src], ranks=data.rank[src])
end

function hutch_ar1_dense_reference(rows, prior, rho, sa, sz, se, eta)
    A = rows.A
    df, dw = vec(sum(A; dims=2)), vec(sum(A; dims=1))
    if prior == BipartiteVarianceStableModel
        Q = [Matrix(Diagonal((1 .+ rho^2 .* (df .- 1)) ./ sa^2)) -rho .* A ./ (sa * sz);
             -rho .* A' ./ (sa * sz) Matrix(Diagonal((1 .+ rho^2 .* (dw .- 1)) ./ sz^2))] ./ (1 - rho^2)
    elseif prior == BipartiteUnnormalizedModel
        Q = [Matrix(Diagonal(df ./ sa^2)) -rho .* A ./ (sa * sz);
             -rho .* A' ./ (sa * sz) Matrix(Diagonal(dw ./ sz^2))]
    else
        W = prior == BipartiteSpectralModel ? A ./ maximum(svdvals(A)) :
            A ./ sqrt.(df * dw')
        Q = [Matrix(Diagonal(fill(inv(sa^2), length(df)))) -rho .* W ./ (sa * sz);
             -rho .* W' ./ (sa * sz) Matrix(Diagonal(fill(inv(sz^2), length(dw))))]
    end
    K = length(rows.y)
    R = [rows.firms[i] == rows.firms[j] ? eta^abs(rows.ranks[i] - rows.ranks[j]) : 0.0
         for i in 1:K, j in 1:K]
    Omega = Symmetric(rows.V * (Q \ rows.V') + se^2 .* R)
    beta = rows.X === nothing ? nothing :
        (rows.X' * (Omega \ rows.X)) \ (rows.X' * (Omega \ rows.y))
    residual = beta === nothing ? rows.y : rows.y - rows.X * beta
    nll = 0.5 * (logdet(Omega) + dot(residual, Omega \ residual))
    M = Q + rows.V' * (R \ rows.V) / se^2
    return (; Q, R, M, beta, nll)
end

function hutch_ar1_stats(data, prior, eta, X; standardize=false)
    return suffstats(prior, data.f, data.w, data.y;
        weighting=Weighting(observations=:raw), standardize,
        error_eta=eta, edge_index=data.rank, match_id=data.match, X)
end

# A sentinel algorithm retains the actual VS model and VSHutchCache dispatch
# while refusing explicit network-precision assembly. This catches the old
# final_mean_profile fallback in a complete initialization-to-finalization fit.
struct HutchAR1NoFactorAlgorithm end
BipartiteGMRF.model_precision(::BipartiteVarianceStableModel{HutchAR1NoFactorAlgorithm},
    ::Float64, ::Float64, ::Float64) =
    error("Network precision assembly/factorization is forbidden in this test")

@testset "HutchSLQ AR(1): dense oracles and factorization-free fitting" begin
    bg = BipartiteGMRF
    data = hutch_ar1_fixture()
    rho, sa, sz, se = 0.2, 0.8, 0.6, 0.7
    limit = 0.8
    seed = 20261007
    solver = HutchSLQ(logdet_probes=1024, lanczos_iters=20,
        cg_tol=1e-12, cg_maxiter=100, optim_iters=4)

    @testset "fixed and estimated eta; all weighted products and cache transitions" begin
        for prior in (BipartiteVarianceStableModel, BipartiteNormalizedModel,
                      BipartiteUnnormalizedModel, BipartiteSpectralModel),
            X in (nothing, data.X, Matrix(data.X))
            rows = hutch_ar1_dense_rows(data, X)
            ss = hutch_ar1_stats(data, prior, :estimate, X)
            model = prior(ss.A_prior; rho_limit=limit)
            cache = bg.make_nll_cache(solver, model, ss)
            exact_cache = prior == BipartiteSpectralModel ? nothing :
                bg.make_nll_cache(ExactCholesky(), model, ss)
            @test ss.K == 6
            @test ss.error_ar1.n_blocks == 3
            @test ss.metadata.graph_only_rows == 3
            @test Matrix(model.graph.A) == rows.A
            @test size(ss.A_prior) == (4, 7)
            first_value = nothing
            for eta in (0.45, 0.0, -0.35, 0.45)
                theta = [atanh(rho / limit), log(sa), log(sz), log(se), atanh(eta)]
                obs = bg.objective_stats(model, ss, theta)
                ref = hutch_ar1_dense_reference(rows, prior, rho, sa, sz, se, eta)
                @test Matrix(obs.design.VtV) ≈ rows.V' * (ref.R \ rows.V) atol=2e-12
                @test obs.design.projected_y ≈ rows.V' * (ref.R \ rows.y) atol=2e-12
                @test obs.design.ydot ≈ dot(rows.y, ref.R \ rows.y) atol=2e-12
                @test -obs.weights.log_weight_sum ≈ (ss.K - 3) * log1p(-eta^2) atol=1e-12
                @test -obs.weights.log_weight_sum ≈ logdet(Symmetric(ref.R)) atol=1e-12
                if X !== nothing
                    @test issparse(obs.mean_stats.VtX) == issparse(X)
                    @test Matrix(obs.mean_stats.VtX) ≈ rows.V' * (ref.R \ rows.X) atol=2e-12
                    @test obs.mean_stats.XtX ≈ rows.X' * (ref.R \ rows.X) atol=2e-12
                    @test obs.mean_stats.Xty ≈ rows.X' * (ref.R \ rows.y) atol=2e-12
                end
                value = bg.nll_value(solver, model, ss, theta, obs, cache; seed)
                if exact_cache !== nothing
                    exact = bg.nll_value(ExactCholesky(), model, ss, theta, obs, exact_cache; seed)
                    @test exact ≈ ref.nll atol=2e-9
                end
                @test isfinite(value) && value < bg.BIG_NLL
                # The stochastic logdet is tested with a finite-probe bound;
                # subtract its measured error to test the remaining likelihood
                # arithmetic at deterministic linear-solve accuracy.
                p = bg.unpack_params(theta; rho_limit=limit)
                ld = bg.hutch_logdet_difference!(cache, solver, length(model), seed, p, inv(se^2))
                ld_exact = logdet(Symmetric(ref.M)) - logdet(Symmetric(ref.Q))
                @test value - 0.5 * (ld - ld_exact) ≈ ref.nll atol=2e-9
                @test abs(value - ref.nll) < 0.45
                @test cache.dV ≈ diag(obs.design.VtV) atol=1e-14
                @test cache.Mdiag ≈ diag(ref.M) atol=2e-12
                if X !== nothing
                    _, beta = bg.final_mean_profile(solver, model, ss, obs, p, cache)
                    @test beta ≈ ref.beta atol=2e-9
                    # Independently owned coefficients survive later PCG calls.
                    saved = copy(beta)
                    bg.pcg_solve!(cache.pcg, cache.mop, ones(length(model));
                        tol=solver.cg_tol, maxiter=solver.cg_maxiter, Mdiag=cache.Mdiag)
                    @test beta == saved
                end
                ss_fixed = hutch_ar1_stats(data, prior, eta, X)
                obs_fixed = bg.objective_stats(model, ss_fixed, theta[1:4])
                fixed_cache = bg.make_nll_cache(solver, model, ss_fixed)
                fixed_value = bg.nll_value(solver, model, ss_fixed, theta[1:4], obs_fixed, fixed_cache; seed)
                @test fixed_value ≈ value atol=2e-11
                if first_value === nothing
                    first_value = value
                elseif eta == 0.45
                    @test value == first_value
                end
                if eta == 0.0
                    ss_iid = suffstats(prior, data.f, data.w, data.y;
                        weighting=Weighting(observations=:raw), standardize=false,
                        match_id=data.match, X)
                    iid_cache = bg.make_nll_cache(solver, model, ss_iid)
                    obs_iid = bg.objective_stats(model, ss_iid, theta[1:4])
                    iid_value = bg.nll_value(solver, model, ss_iid, theta[1:4], obs_iid, iid_cache; seed)
                    @test iid_value ≈ value atol=2e-11
                end
            end
        end
    end

    @testset "ungrouped raw observations" begin
        raw = merge(data, (match=collect(eachindex(data.y)),
            rank=[1, 2, 3, 4, 1, 2, 1, 99, 99, 99]))
        for X in (nothing, raw.X), eta_spec in (0.35, :estimate)
            rows = hutch_ar1_dense_rows(raw, X)
            ss = suffstats(BipartiteVarianceStableModel, raw.f, raw.w, raw.y;
                weighting=Weighting(observations=:raw), standardize=false,
                error_eta=eta_spec, edge_index=raw.rank, X)
            model = BipartiteVarianceStableModel(ss.A_prior; rho_limit=limit)
            eta = eta_spec === :estimate ? -0.25 : eta_spec
            theta = [atanh(rho / limit), log(sa), log(sz), log(se)]
            eta_spec === :estimate && push!(theta, atanh(eta))
            obs = bg.objective_stats(model, ss, theta)
            cache = bg.make_nll_cache(solver, model, ss)
            ref = hutch_ar1_dense_reference(rows, BipartiteVarianceStableModel, rho, sa, sz, se, eta)
            @test ss.K == 7
            @test ss.error_ar1.n_blocks == 3
            @test Matrix(obs.design.VtV) ≈ rows.V' * (ref.R \ rows.V) atol=2e-12
            exact = bg.nll_value(ExactCholesky(), model, ss, theta, obs,
                bg.make_nll_cache(ExactCholesky(), model, ss); seed)
            value = bg.nll_value(solver, model, ss, theta, obs, cache; seed)
            @test exact ≈ ref.nll atol=2e-9
            @test abs(value - ref.nll) < 0.45
            if X !== nothing
                _, beta = bg.final_mean_profile(solver, model, ss, obs,
                    bg.unpack_params(theta; rho_limit=limit), cache)
                @test beta ≈ ref.beta atol=2e-9
            end
            result = fit_mle(model, ss; solver=HutchSLQ(logdet_probes=32,
                lanczos_iters=20, cg_tol=1e-11, cg_maxiter=100, optim_iters=3),
                fix_rho=rho, init=(sigma_a=sa, sigma_z=sz, sigma_epsilon=se, eta), seed)
            @test isfinite(result.nll) && result.nll < bg.BIG_NLL
            @test (result.beta === nothing) == (X === nothing)
            @test loglikelihood(result) ≈ -(result.nll + 3.5log(2pi)) atol=1e-11
        end
    end

    @testset "same sparse object, new eta values" begin
        ss = hutch_ar1_stats(data, BipartiteVarianceStableModel, :estimate, data.X)
        model = BipartiteVarianceStableModel(ss.A_prior; rho_limit=limit)
        theta0 = [atanh(rho / limit), log(sa), log(sz), log(se), atanh(0.4)]
        obs0 = bg.objective_stats(model, ss, theta0)
        cache = bg.make_nll_cache(solver, model, ss)
        bg.nll_value(solver, model, ss, theta0, obs0, cache; seed)
        shared = obs0.design.VtV
        for eta in (0.0, -0.5, 0.3)
            theta = vcat(theta0[1:4], atanh(eta))
            target = bg.objective_stats(model, ss, theta)
            @test shared.colptr == target.design.VtV.colptr
            @test shared.rowval == target.design.VtV.rowval
            copyto!(shared.nzval, target.design.VtV.nzval)
            d = target.design
            changed = bg.DesignStats(shared, d.projected_y, d.ydot, d.A_obs, d.At_obs, d.FF, d.WW)
            obs = bg.ObservationStats(changed, target.weights, target.rho_eps, target.mean_stats)
            @test cache.mop.VtV === shared
            value = bg.nll_value(solver, model, ss, theta, obs, cache; seed)
            fresh = bg.make_nll_cache(solver, model, ss)
            reference = bg.nll_value(solver, model, ss, theta, target, fresh; seed)
            @test value ≈ reference atol=2e-11
            @test cache.dV == diag(shared)
        end
    end

    @testset "grouping invariance, constants, and iterative final coefficients" begin
        small = HutchSLQ(logdet_probes=32, lanczos_iters=20,
            cg_tol=1e-11, cg_maxiter=100, optim_iters=4)
        order = [7, 5, 2, 10, 4, 6, 1, 9, 3, 8]
        shuffled = (; (name => (name == :X ? data.X[order, :] : getproperty(data, name)[order])
                         for name in propertynames(data))...)
        duplicated = (f=vcat(data.f, data.f[2]), w=vcat(data.w, data.w[2]),
            y=vcat(data.y, data.y[2]), match=vcat(data.match, data.match[2]),
            rank=vcat(data.rank, data.rank[2]), X=vcat(data.X, data.X[2:2, :]))
        for X_present in (false, true), eta_spec in (0.4, -0.3, :estimate),
            fix_rho in (nothing, rho)
            init = (rho=rho, sigma_a=sa, sigma_z=sz, sigma_epsilon=se,
                    eta=eta_spec === :estimate ? -0.15 : eta_spec)
            outcomes = map((data, shuffled, duplicated)) do current
                ss = hutch_ar1_stats(current, BipartiteVarianceStableModel, eta_spec,
                    X_present ? current.X : nothing; standardize=true)
                fit_mle(BipartiteVarianceStableModel, ss;
                    solver=small, rho_limit=limit, fix_rho, init, seed)
            end
            result = first(outcomes)
            @test isfinite(result.nll) && result.nll < bg.BIG_NLL
            @test -1 < result.eta < 1
            @test result.stats.K == 6
            @test bg.rho_limit(result.model) == limit
            @test Matrix(result.model.graph.A) == hutch_ar1_dense_rows(data, nothing).A
            @test fix_rho === nothing || isapprox(result.rho, fix_rho; atol=1e-15)
            @test eta_spec === :estimate || result.eta == eta_spec
            @test loglikelihood(result) ≈ -(result.nll + 6log(result.stats.y_std) + 3log(2pi)) atol=1e-11
            for other in outcomes[2:end]
                @test other.nll ≈ result.nll atol=1e-10
                @test coef(other) ≈ coef(result) atol=1e-10
                @test loglikelihood(other) ≈ loglikelihood(result) atol=1e-10
            end
            if X_present
                rows = hutch_ar1_dense_rows(data, data.X)
                centered = merge(rows, (y=(rows.y .- result.stats.y_mean) ./ result.stats.y_std,))
                p = bg.scaled_params(result)
                ref = hutch_ar1_dense_reference(centered, BipartiteVarianceStableModel,
                    p.rho, p.sigma_a, p.sigma_z, p.sigma_epsilon, result.eta)
                @test result.beta ≈ ref.beta .* result.stats.y_std atol=2e-8
            else
                @test result.beta === nothing
            end
        end
        # An actual fit with controls, rather than just a finalization unit test.
        ss = hutch_ar1_stats(data, BipartiteVarianceStableModel, :estimate, data.X)
        nofactor = BipartiteVarianceStableModel(ss.A_prior;
            rho_limit=limit, alg=HutchAR1NoFactorAlgorithm())
        @test bg.make_nll_cache(small, nofactor, ss) isa bg.VSHutchCache
        result = fit_mle(nofactor, ss; solver=small, fix_rho=rho,
            init=(sigma_a=sa, sigma_z=sz, sigma_epsilon=se, eta=0.2), seed)
        @test result.beta !== nothing
        @test all(isfinite, result.beta)
        @test isfinite(result.nll) && result.nll < bg.BIG_NLL
        @test_throws ArgumentError decompose(result; kind=:model, probes=2)
        @test_throws ArgumentError decompose(result; kind=:fitted, probes=2)
    end

    @testset "failed iterative solves cannot become successful fits" begin
        bad = HutchSLQ(logdet_probes=2, lanczos_iters=4,
            cg_tol=1e-15, cg_maxiter=1, optim_iters=2)
        for X in (nothing, data.X)
            ss = hutch_ar1_stats(data, BipartiteVarianceStableModel, :estimate, X)
            model = BipartiteVarianceStableModel(ss.A_prior; rho_limit=limit)
            theta = [atanh(rho / limit), log(sa), log(sz), log(se), atanh(0.4)]
            obs = bg.objective_stats(model, ss, theta)
            cache = bg.make_nll_cache(bad, model, ss)
            @test bg.nll_value(bad, model, ss, theta, obs, cache; seed) == bg.BIG_NLL
            error_type = X === nothing ? ErrorException : bg.MeanProfileError
            @test_throws error_type fit_mle(model, ss; solver=bad, fix_rho=rho,
                init=(sigma_a=sa, sigma_z=sz, sigma_epsilon=se, eta=0.4), seed)
            if X !== nothing
                decoded = bg.unpack_params(theta; rho_limit=limit)
                @test_throws bg.MeanProfileError bg.final_mean_profile(bad, model, ss, obs, decoded, cache)
            end
        end
        # The convergence flag is checked against a freshly computed residual,
        # not only CG's recursively updated residual vector.
        A = Symmetric([1e-4 2e-5 0.0; 2e-5 1.0 0.1; 0.0 0.1 10.0])
        b = [1.0, -2.0, 0.5]
        ws = bg.PCGWorkspace(3)
        x, ok, _, relres = bg.pcg_solve!(ws, (out, v) -> mul!(out, A, v), b;
            tol=1e-11, maxiter=100, Mdiag=diag(A))
        @test ok
        @test norm(A * x - b) / norm(b) <= 1e-11
        @test relres ≈ norm(A * x - b) / norm(b) atol=1e-15
    end

    @testset "unresolved eta boundaries and nonfinite trials are rejected" begin
        fast = HutchSLQ(logdet_probes=4, lanczos_iters=20,
            cg_tol=1e-13, cg_maxiter=300, optim_iters=1,
            simplex_scale=0.0, simplex_shift=0.01)
        covariance_codes = [atanh(rho / limit), log(sa), log(sz), log(se)]
        for X in (nothing, data.X)
            ss = hutch_ar1_stats(data, BipartiteVarianceStableModel, :estimate, X)
            model = BipartiteVarianceStableModel(ss.A_prior; rho_limit=limit)
            safe_theta = vcat(covariance_codes, atanh(0.3))
            safe_obs = bg.objective_stats(model, ss, safe_theta)
            cache = bg.make_nll_cache(fast, model, ss)
            # Rejected before statistic assembly in optimization; direct NLL
            # callers are also guarded, even if they supply existing stats.
            for coordinate in (-1e300, -20.0, 20.0, 1e300, -Inf, Inf, NaN)
                theta = vcat(covariance_codes, coordinate)
                @test !bg.objective_parameters_valid(fast, ss, theta)
                @test bg.nll_value(fast, model, ss, theta, safe_obs, cache; seed) == bg.BIG_NLL
            end
            for eta in (-0.99, 0.99)
                theta = vcat(covariance_codes, atanh(eta))
                @test bg.objective_parameters_valid(fast, ss, theta)
                obs = bg.objective_stats(model, ss, theta)
                value = bg.nll_value(fast, model, ss, theta, obs, cache; seed)
                @test isfinite(value) && value < bg.BIG_NLL
            end
            for eta in (-prevfloat(1.0), prevfloat(1.0))
                fixed = hutch_ar1_stats(data, BipartiteVarianceStableModel, eta, X)
                fixed_cache = bg.make_nll_cache(fast, model, fixed)
                @test !bg.objective_parameters_valid(fast, fixed, covariance_codes)
                @test bg.objective_parameters_valid(ExactCholesky(), fixed, covariance_codes)
                @test bg.nll_value(fast, model, fixed, covariance_codes,
                    safe_obs, fixed_cache; seed) == bg.BIG_NLL
                error_type = X === nothing ? ErrorException : bg.MeanProfileError
                for current in (fixed, ss)
                    failure = try
                        fit_mle(model, current; solver=fast, fix_rho=rho,
                            init=(sigma_a=sa, sigma_z=sz, sigma_epsilon=se, eta), seed)
                        nothing
                    catch err
                        err
                    end
                    @test failure isa error_type
                    @test occursin("AR(1) eta is too close", sprint(showerror, failure))
                    @test occursin("No fitted result", sprint(showerror, failure))
                end
            end
            # All solver paths reject nonfinite coordinates before decoding,
            # independently of whether eta is fixed or estimated.
            for coordinate in (-Inf, Inf, NaN), solver_kind in (fast, ExactCholesky())
                theta = copy(safe_theta)
                theta[2] = coordinate
                @test !bg.objective_parameters_valid(solver_kind, ss, theta)
                theta[2] = safe_theta[2]
                theta[5] = coordinate
                @test !bg.objective_parameters_valid(solver_kind, ss, theta)
            end
        end
    end

    @testset "unsupported combinations stay rejected" begin
        for weighting in (Weighting(observations=:edge), Weighting(observations=:effective, rho_eps=0.2))
            @test_throws ArgumentError suffstats(BipartiteNormalizedModel, data.f, data.w, data.y;
                weighting, error_eta=0.3, edge_index=data.rank)
        end
        ss = hutch_ar1_stats(data, BipartiteVarianceStableModel, 0.3, nothing)
        model = BipartiteVarianceStableModel(ss.A_prior; rho_limit=limit)
        @test_throws ArgumentError bg.validate_capability(model, ss, EMIWBlocks())
    end
end
