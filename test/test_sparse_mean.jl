using Serialization: serialize, deserialize

# Small observation-space oracles deliberately avoid package mean-stat
# builders and the profiling kernel under test.
function sparse_mean_rows(f, w, y, X; match_id=nothing, nf=maximum(f), nw=maximum(w))
    observed = findall(isfinite, y)
    groups = match_id === nothing ? [[i] for i in observed] :
        [filter(i -> match_id[i] == m, observed) for m in unique(match_id[observed])]
    V = zeros(length(groups), nf + nw)
    src = [first(rows) for rows in groups]
    for (j, rows) in enumerate(groups)
        firms, workers = unique(f[rows]), unique(w[rows])
        V[j, firms] .= inv(length(firms))
        V[j, nf .+ workers] .= inv(length(workers))
    end
    return V, y[src], X === nothing ? nothing : Matrix(X[src, :]), src
end

function sparse_mean_precision(model, rho, sa, sz)
    A = Matrix(model.graph.A)
    df, dw = vec(sum(A; dims=2)), vec(sum(A; dims=1))
    if model isa BipartiteVarianceStableModel
        return [Matrix(Diagonal((1 .+ rho^2 .* (df .- 1)) ./ sa^2)) -rho .* A ./ (sa*sz);
                -rho .* A' ./ (sa*sz) Matrix(Diagonal((1 .+ rho^2 .* (dw .- 1)) ./ sz^2))] ./ (1-rho^2)
    elseif model isa BipartiteUnnormalizedModel
        return [Matrix(Diagonal(df ./ sa^2)) -rho .* A ./ (sa*sz);
                -rho .* A' ./ (sa*sz) Matrix(Diagonal(dw ./ sz^2))]
    else
        W = A ./ sqrt.(df * dw')
        return [Matrix(Diagonal(fill(inv(sa^2), length(df)))) -rho .* W ./ (sa*sz);
                -rho .* W' ./ (sa*sz) Matrix(Diagonal(fill(inv(sz^2), length(dw))))]
    end
end

function sparse_mean_oracle(model, V, y, X, R, rho, sa, sz, se)
    Q = sparse_mean_precision(model, rho, sa, sz)
    Omega = Symmetric(V * (Q \ V') + se^2 * R)
    P = V' * (R \ V)
    M = Q + P / se^2
    if X === nothing
        beta, correction = nothing, 0.0
    else
        G = Symmetric(X' * (Omega \ X))
        c = X' * (Omega \ y)
        beta = G \ c
        correction = dot(c, beta)
    end
    nll = 0.5 * (logdet(Omega) + dot(y, Omega \ y) - correction)
    return (; Q, M, beta, correction, nll)
end

function sparse_mean_check(model, ss, V, y, X, R;
    rho=0.2, sa=0.8, sz=0.6, se=0.5, extra=Float64[], cache=nothing)
    bg = BipartiteGMRF
    theta = vcat([atanh(rho / bg.rho_limit(model)), log(sa), log(sz), log(se)], extra)
    obs = bg.objective_stats(model, ss, theta)
    ew = cache === nothing ? bg.make_exact_workspace(model, ss) : cache
    ref = sparse_mean_oracle(model, V, y, X, R, rho, sa, sz, se)
    value = bg.nll_exact_value(model, ss, theta, obs, ew)
    @test isfinite(value) && value < bg.BIG_NLL
    @test value ≈ ref.nll atol=1e-9 rtol=1e-9
    if X === nothing
        @test obs.mean_stats === nothing
        @test ew.mean === nothing
        return (; theta, obs, cache=ew, ref)
    end
    ms = obs.mean_stats
    @test Matrix(ms.VtX) ≈ V' * (R \ X) atol=1e-11 rtol=1e-11
    @test ms.XtX ≈ X' * (R \ X) atol=1e-11 rtol=1e-11
    @test ms.Xty ≈ X' * (R \ y) atol=1e-11 rtol=1e-11
    @test ew.mean isa bg.MeanProfileWorkspace
    factor = cholesky(Symmetric(ref.M))
    scratch = zeros(size(V, 2))
    calls = Ref(0)
    function borrowed(v)
        calls[] += 1
        return copyto!(scratch, factor \ v)
    end
    for block_size in (1, 2, ms.p + 2)
        workspace = bg.MeanProfileWorkspace(ms; block_size)
        @test size(workspace.solved) == (size(V, 2), min(block_size, ms.p))
        # Deliberately dirty every work buffer, including the final short
        # block, to detect accidental reuse of stale values.
        for name in (:rhs, :solved, :cross, :G, :c)
            fill!(getfield(workspace, name), NaN)
        end
        calls[] = 0
        correction, beta = bg.mean_profile_correction(
            ms, inv(se^2), obs.design.projected_y, borrowed, workspace)
        @test calls[] == ms.p + 1
        @test beta ≈ ref.beta atol=1e-9 rtol=1e-9
        @test correction ≈ ref.correction atol=1e-9 rtol=1e-9
        saved_beta = copy(beta)
        solved_y = borrowed(obs.design.projected_y)
        calls[] = 0
        correction2, beta2 = bg.mean_profile_correction(
            ms, inv(se^2), obs.design.projected_y, borrowed, workspace; solved_y)
        @test calls[] == ms.p
        @test beta2 ≈ ref.beta atol=1e-9 rtol=1e-9
        @test correction2 ≈ ref.correction atol=1e-9 rtol=1e-9
        fill!(workspace.c, -100.0)
        fill!(workspace.solved, -100.0)
        @test beta == saved_beta
        @test beta2 ≈ saved_beta atol=1e-9 rtol=1e-9
    end
    return (; theta, obs, cache=ew, ref)
end

@testset "sparse and bounded-workspace mean profiling" begin
    bg = BipartiteGMRF
    f = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    w = [1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4]
    y = [0.4, 1.2, -0.3, 0.8, 1.1, -0.6, 0.2, 0.9, -0.4, 1.4, 0.5, -0.2]
    K = length(y)
    t = collect(range(-1.0, 1.0; length=K))
    category = mod1.(1:K, 3)
    indicators = Matrix(sparse(1:K, category, ones(K), K, 3))
    fractional = 0.25 .* indicators + 0.75 .* indicators[:, [3, 1, 2]]
    designs = (
        intercept=ones(K, 1),
        continuous=hcat(ones(K), t, t.^2),
        indicators=indicators,
        fractional=fractional,
        mixed=hcat(ones(K), t, indicators[:, 2]),
        interactions=hcat(ones(K), t, indicators[:, 2], t .* indicators[:, 2]),
    )

    @testset "iid dense GLS, design types, and block sizes" begin
        for prior in (BipartiteNormalizedModel, BipartiteUnnormalizedModel,
                      BipartiteVarianceStableModel)
            ss0 = suffstats(prior, f, w, y; standardize=false)
            model = prior(ss0.A_prior; rho_limit=0.8)
            V, yo, _, _ = sparse_mean_rows(f, w, y, nothing)
            R = Matrix{Float64}(I, K, K)
            sparse_mean_check(model, ss0, V, yo, nothing, R)
            for (name, Xd) in pairs(designs), storage in (:dense, :sparse)
                @testset "$prior / $name / $storage" begin
                    X = storage == :sparse ? sparse(Xd) : Xd
                    ss = suffstats(prior, f, w, y; standardize=false, X=X)
                    @test (ss.mean_stats.VtX isa SparseMatrixCSC) == (storage == :sparse)
                    sparse_mean_check(model, ss, V, yo, Xd, R)
                end
            end
        end
        for Xreal in (SparseMatrixCSC{Float32,Int32}(sparse(designs.fractional)),
                      sparse(Int.(indicators)), sparse(Bool.(indicators)))
            ssreal = suffstats(BipartiteNormalizedModel, f, w, y;
                standardize=false, X=Xreal)
            @test ssreal.mean_stats.VtX isa SparseMatrixCSC{Float64,Int}
            modelreal = BipartiteNormalizedModel(ssreal.A_prior; rho_limit=0.8)
            V, yo, Xo, _ = sparse_mean_rows(f, w, y, Xreal)
            sparse_mean_check(modelreal, ssreal, V, yo, Float64.(Xo), Matrix{Float64}(I, K, K))
        end
    end

    @testset "match AR1 components, cancellation, and finite differences" begin
        fm = [1, 1, 1, 1, 2, 2, 3]
        wm = [1, 2, 1, 3, 4, 4, 5]
        ym = [0.3, 0.3, 1.2, -0.2, 0.7, -0.5, 0.4]
        mid = [10, 10, 20, 30, 40, 50, 60]
        rank = [1, 1, 2, 3, 1, 2, 1]
        Xm = [1.0 0 0; 1 0 0; 0 1 0; 0 0 1; 1 0 0; 0 1 0; 0 0 1]
        V, yo, Xo, src = sparse_mean_rows(fm, wm, ym, Xm; match_id=mid)
        Reta(eta) = [fm[src[i]] == fm[src[j]] ? eta^abs(rank[src[i]]-rank[src[j]]) : 0.0
                     for i in eachindex(src), j in eachindex(src)]
        for prior in (BipartiteNormalizedModel, BipartiteVarianceStableModel),
            storage in (:dense, :sparse)
            X = storage == :sparse ? sparse(Xm) : Xm
            ss = suffstats(prior, fm, wm, ym; X=X, standardize=false,
                match_id=mid, error_eta=:estimate, edge_index=rank)
            model = prior(ss.A_prior; rho_limit=0.8)
            for component in (ss.error_ar1.mean_full, ss.error_ar1.mean_adj, ss.error_ar1.mean_int)
                @test (component.VtX isa SparseMatrixCSC) == (storage == :sparse)
            end
            cache = bg.make_exact_workspace(model, ss)
            mean_workspace = cache.mean
            factor_before = cache.ws_M.backend.factor
            for eta in (0.3, 0.0, -0.4, 0.3)
                out = sparse_mean_check(model, ss, V, yo, Xo, Reta(eta);
                    extra=[atanh(eta)], cache=cache)
                @test (out.obs.mean_stats.VtX isa SparseMatrixCSC) == (storage == :sparse)
                @test cache.mean === mean_workspace
                @test cache.ws_M.backend.factor === factor_before
            end
            theta = [atanh(0.2/0.8), log(0.8), log(0.6), log(0.5), atanh(0.3)]
            exact_value(z) = bg.nll_exact_value(model, ss, z,
                bg.objective_stats(model, ss, z), cache)
            oracle_value(z) = sparse_mean_oracle(model, V, yo, Xo, Reta(tanh(z[5])),
                0.8*tanh(z[1]), exp(z[2]), exp(z[3]), exp(z[4])).nll
            for j in (1, 5)
                h = 1e-5
                plus, minus = copy(theta), copy(theta)
                plus[j] += h; minus[j] -= h
                @test (exact_value(plus)-exact_value(minus))/(2h) ≈
                    (oracle_value(plus)-oracle_value(minus))/(2h) atol=1e-6 rtol=1e-6
            end
            invalid = copy(theta); invalid[2] = Inf
            @test exact_value(invalid) == bg.BIG_NLL
            @test exact_value(theta) ≈ oracle_value(theta) atol=1e-9 rtol=1e-9
            for eta in (0.0, -0.3)
                fixed = suffstats(prior, fm, wm, ym; X=X, standardize=false,
                    match_id=mid, error_eta=eta, edge_index=rank)
                sparse_mean_check(model, fixed, V, yo, Xo, Reta(eta))
            end
        end
        # Also exercise sparse mean preparation when each raw row is its
        # own AR1 observation, including repeated firm-worker edges.
        ranks = repeat(1:4; outer=3)
        raw = suffstats(BipartiteNormalizedModel, f, w, y; X=sparse(designs.mixed),
            standardize=false, error_eta=:estimate, edge_index=ranks)
        rawmodel = BipartiteNormalizedModel(raw.A_prior; rho_limit=0.8)
        Vr, yr, Xr, _ = sparse_mean_rows(f, w, y, designs.mixed)
        for eta in (0.0, -0.3)
            R = [f[i] == f[j] ? eta^abs(ranks[i]-ranks[j]) : 0.0 for i in 1:K, j in 1:K]
            sparse_mean_check(rawmodel, raw, Vr, yr, Xr, R; extra=[atanh(eta)])
        end
    end

    @testset "first member controls are preserved, not averaged" begin
        fm, wm = [1, 1, 1, 2, 2], [1, 2, 3, 2, 4]
        ym, mid, rank = [0.3, 0.3, 1.2, 0.7, -0.5], [1, 1, 2, 3, 4], [1, 1, 2, 1, 2]
        Xbase = hcat(ones(5), [0.0, 7.0, 1.0, -1.0, 2.0])
        for error_model in (:iid, :ar1, :correlated), storage in (:dense, :sparse)
            likelihoods = Float64[]
            for perm in ([1, 2, 3, 4, 5], [2, 1, 3, 4, 5])
                fp, wp, yp, mp, rp = fm[perm], wm[perm], ym[perm], mid[perm], rank[perm]
                X = storage == :sparse ? sparse(Xbase[perm, :]) : Xbase[perm, :]
                kwargs = error_model == :ar1 ? (; error_eta=0.3, edge_index=rp) :
                    error_model == :correlated ? (; error_cov=sparse(Matrix{Float64}(I, 5, 5))) : (;)
                ss = suffstats(BipartiteNormalizedModel, fp, wp, yp;
                    standardize=false, X=X, match_id=mp, kwargs...)
                model = BipartiteNormalizedModel(ss.A_prior; rho_limit=0.8)
                V, yo, Xo, src = sparse_mean_rows(fp, wp, yp, X; match_id=mp)
                @test Xo[1, 2] == Xbase[first(perm), 2]
                R = error_model == :ar1 ? [fp[src[i]] == fp[src[j]] ? 0.3^abs(rp[src[i]]-rp[src[j]]) : 0.0
                           for i in eachindex(src), j in eachindex(src)] : Matrix{Float64}(I, 4, 4)
                out = sparse_mean_check(model, ss, V, yo, Xo, R)
                push!(likelihoods, out.ref.nll)
            end
            @test abs(likelihoods[1]-likelihoods[2]) > 1e-4
        end
    end

    @testset "existing correlated and group-mean error paths" begin
        Xd = designs.mixed[:, 1:2]
        V, yo, _, _ = sparse_mean_rows(f, w, y, nothing)
        Rin = Matrix(Diagonal(1.0 .+ 0.02 .* (1:K)))
        for i in 1:K, j in 1:K
            i != j && f[i] == f[j] && (Rin[i, j] = 0.1)
        end
        R = Rin .* (K/tr(Rin))
        groups = [1, 1, 2, 2, 2, 3, 4, 4, 4, 4, 5, 5]
        ids = unique(groups)
        averaging = zeros(length(ids), K)
        for (i, g) in enumerate(ids)
            rows = findall(==(g), groups)
            averaging[i, rows] .= inv(length(rows))
        end
        omega = Dict(1=>1.0, 2=>1.3, 3=>0.7, 4=>1.8)
        Rg = Matrix(Diagonal([omega[count(==(g), groups)] for g in ids]))
        for storage in (:dense, :sparse)
            X = storage == :sparse ? sparse(Xd) : Xd
            ss = suffstats(BipartiteNormalizedModel, f, w, y;
                standardize=false, X=X, error_cov=sparse(Rin))
            model = BipartiteNormalizedModel(ss.A_prior; rho_limit=0.8)
            @test (ss.mean_stats.VtX isa SparseMatrixCSC) == (storage == :sparse)
            sparse_mean_check(model, ss, V, yo, Xd, R)
            sg = suffstats(BipartiteNormalizedModel, f, w, y;
                standardize=false, X=X, error_groups=groups)
            for component in sg.error_classes.mean_stats
                @test (component.VtX isa SparseMatrixCSC) == (storage == :sparse)
            end
            out = sparse_mean_check(model, sg, averaging*V, averaging*yo,
                averaging*Xd, Rg; extra=log.([1.3, 0.7, 1.8]))
            @test (out.obs.mean_stats.VtX isa SparseMatrixCSC) == (storage == :sparse)
        end
    end

    @testset "effective-row validation and graph-only rows" begin
        for storage in (:dense, :sparse)
            convert_X(A) = storage == :sparse ? sparse(A) : A
            for invalid in (NaN, Inf)
                X = copy(designs.mixed); X[1, 2] = invalid
                @test_throws ArgumentError suffstats(BipartiteNormalizedModel,
                    f, w, y; X=convert_X(X))
            end
            @test_throws ArgumentError suffstats(BipartiteNormalizedModel,
                f, w, y; X=convert_X(hcat(ones(K), zeros(K))))
            @test_throws ArgumentError suffstats(BipartiteNormalizedModel,
                f, w, y; X=convert_X(ones(K, K+1)))
            @test_throws ArgumentError suffstats(BipartiteNormalizedModel,
                [1, 1, 2], [1, 2, 3], [0.5, 0.5, 0.2];
                match_id=[1, 1, 2], X=convert_X(ones(3, 3)))
            Xgo = convert_X(vcat(designs.mixed, fill(NaN, 1, 3)))
            fg, wg, yg = vcat(f, 4), vcat(w, 5), vcat(y, NaN)
            ss = suffstats(BipartiteNormalizedModel, fg, wg, yg; X=Xgo, standardize=false)
            @test ss.metadata.graph_only_rows == 1
            @test ss.K == K
            model = BipartiteNormalizedModel(ss.A_prior; rho_limit=0.8)
            V, yo, Xo, _ = sparse_mean_rows(fg, wg, yg, Xgo)
            sparse_mean_check(model, ss, V, yo, Xo, Matrix{Float64}(I, K, K))
            # A nonfirst match member is not the effective control row.
            ignored = convert_X(reshape([1.0, NaN, 2.0], 3, 1))
            selected = suffstats(BipartiteNormalizedModel, [1, 1, 2], [1, 2, 3],
                [0.5, 0.5, 0.2]; match_id=[1, 1, 2], X=ignored, standardize=false)
            @test selected.mean_stats.XtX ≈ reshape([5.0], 1, 1)
        end
    end

    @testset "supported HutchSLQ sparse and dense mean paths" begin
        Xd = designs.mixed
        V, yo, _, _ = sparse_mean_rows(f, w, y, nothing)
        Rin = Matrix(Diagonal(1.0 .+ 0.02 .* (1:K)))
        for i in 1:K, j in 1:K
            i != j && f[i] == f[j] && (Rin[i, j] = 0.1)
        end
        solver = HutchSLQ(logdet_probes=8, lanczos_iters=size(V, 2),
            cg_tol=1e-13, cg_maxiter=100, optim_iters=1)
        for prior in (BipartiteNormalizedModel, BipartiteVarianceStableModel),
            error_model in (:iid, :singleton_groups, :correlated)
            kwargs = error_model == :singleton_groups ? (; error_groups=collect(1:K)) :
                error_model == :correlated ? (; error_cov=sparse(Rin)) : (;)
            R = error_model == :correlated ? Rin .* (K/tr(Rin)) : Matrix{Float64}(I, K, K)
            ss0 = suffstats(prior, f, w, y; standardize=false, kwargs...)
            ssd = suffstats(prior, f, w, y; X=Xd, standardize=false, kwargs...)
            sss = suffstats(prior, f, w, y; X=sparse(Xd), standardize=false, kwargs...)
            model = prior(ss0.A_prior; rho_limit=0.8)
            for stats in (ss0, ssd, sss)
                @test bg.validate_capability(model, stats, solver) === nothing
            end
            caches = map(stats -> bg.make_hutch_cache(model, stats, solver), (ss0, ssd, sss))
            @test caches[1].mean === nothing
            @test caches[2].mean isa bg.MeanProfileWorkspace
            @test caches[3].mean isa bg.MeanProfileWorkspace
            retained = (caches[2].mean, caches[3].mean)
            for (rho, se, seed) in ((0.2, 0.5, 17), (-0.15, 0.65, 42), (0.2, 0.5, 17))
                theta = [atanh(rho/0.8), log(0.8), log(0.6), log(se)]
                observations = map(stats -> bg.objective_stats(model, stats, theta), (ss0, ssd, sss))
                values = map((stats, obs, cache) -> bg.nll_hutch_value(
                    model, stats, solver, theta, obs, cache; seed=seed),
                    (ss0, ssd, sss), observations, caches)
                @test all(isfinite, values)
                @test maximum(values) < bg.BIG_NLL
                @test values[2] ≈ values[3] atol=1e-10 rtol=1e-10
                @test caches[2].mean === retained[1]
                @test caches[3].mean === retained[2]
                ref = sparse_mean_oracle(model, V, yo, Xd, R, rho, 0.8, 0.6, se)
                # Paired probe streams cancel stochastic logdet error; this
                # verifies the real objective's correction, not SLQ accuracy.
                @test 2*(values[1]-values[2]) ≈ ref.correction atol=1e-9 rtol=1e-9
                @test 2*(values[1]-values[3]) ≈ ref.correction atol=1e-9 rtol=1e-9
                cache, obs = caches[3], observations[3]
                function borrowed_pcg(v)
                    solution, ok, _, _ = bg.pcg_solve!(cache.pcg, cache.mop, v;
                        tol=solver.cg_tol, maxiter=solver.cg_maxiter, Mdiag=cache.Mdiag)
                    ok || error("Tiny sparse-mean reference PCG did not converge")
                    return solution
                end
                correction, beta = bg.mean_profile_correction(obs.mean_stats, inv(se^2),
                    obs.design.projected_y, borrowed_pcg, cache.mean)
                @test correction ≈ ref.correction atol=1e-9 rtol=1e-9
                @test beta ≈ ref.beta atol=1e-9 rtol=1e-9
            end
        end
    end

    @testset "bounded storage and explicit numerical failures" begin
        n, p = 200, 30
        B = sparse(1:p, 1:p, fill(0.1, p), n, p)
        ms = bg.MeanStats(B, Matrix{Float64}(I, p, p), ones(p), p)
        workspace = bg.MeanProfileWorkspace(ms)
        @test size(workspace.solved) == (n, 8)
        @test size(workspace.cross) == (p, 8)
        @test Base.summarysize(workspace) < 8*n*p
        @test_throws ArgumentError bg.MeanProfileWorkspace(ms; block_size=0)
        @test_throws DimensionMismatch bg.mean_profile_correction(ms, 1.0,
            zeros(n-1), identity, workspace)

        ms2 = bg.MeanStats(sparse(Matrix{Float64}(I, 2, 2)),
            2 .* Matrix{Float64}(I, 2, 2), ones(2), 2)
        ws2 = bg.MeanProfileWorkspace(ms2; block_size=1)
        @test_throws bg.MeanProfileError bg.mean_profile_correction(ms2, 1.0,
            ones(2), v -> [0.2 0.1; 0.0 0.3] * v, ws2)
        @test_throws bg.MeanProfileError bg.mean_profile_correction(ms2, 1.0,
            ones(2), v -> fill(NaN, 2), ws2)
        correction, beta = bg.mean_profile_correction(ms2, 1.0, ones(2), v -> 0.5 .* v, ws2)
        @test beta ≈ fill(1/3, 2)
        @test correction ≈ 1/3
        singular = suffstats(BipartiteNormalizedModel, f, w, y;
            X=sparse(ones(K, 2)), standardize=false)
        model = BipartiteNormalizedModel(singular.A_prior; rho_limit=0.8)
        theta = [atanh(0.2/0.8), log(0.8), log(0.6), log(0.5)]
        obs = bg.objective_stats(model, singular, theta)
        Q = sparse_mean_precision(model, 0.2, 0.8, 0.6)
        factor = cholesky(Symmetric(Q + 4 .* Matrix(obs.design.VtV)))
        @test_throws bg.MeanProfileError bg.mean_profile_correction(
            obs.mean_stats, 4.0, obs.design.projected_y, v -> factor \ v)
        # Rank failure must not become a fitted result with BIG_NLL or an
        # arbitrary coefficient vector after the optimizer stops.
        for solver in (ExactCholesky(optim_iters=1, polish=false),
                       HutchSLQ(logdet_probes=4, lanczos_iters=7,
                           cg_tol=1e-12, cg_maxiter=100, optim_iters=1))
            @test_throws bg.MeanProfileError fit_mle(BipartiteNormalizedModel,
                singular; solver=solver, seed=17)
        end
        illscaled = bg.MeanStats(zeros(2, 2), [1.0 0; 0 1e-40], ones(2), 2)
        @test_throws bg.MeanProfileError bg.mean_profile_correction(
            illscaled, 1.0, ones(2), identity)

        # Warm each specialization before measuring. These byte bounds are
        # below even one dense K-by-p (preparation) or n-by-p (kernel) array,
        # but leave ample room for the intended sparse and p-by-p work.
        alloc_k, alloc_p = 2000, 200
        alloc_f = collect(1:alloc_k)
        alloc_w, alloc_y = copy(alloc_f), sin.(Float64.(alloc_f))
        alloc_X = sparse(alloc_f, mod1.(alloc_f, alloc_p), ones(alloc_k), alloc_k, alloc_p)
        alloc_ms = bg.build_mean_stats(alloc_f, alloc_w, alloc_y, alloc_X, alloc_k, alloc_k)
        prep_bytes = @allocated bg.build_mean_stats(
            alloc_f, alloc_w, alloc_y, alloc_X, alloc_k, alloc_k)
        @test prep_bytes < sizeof(Float64)*alloc_k*alloc_p
        alloc_ws = bg.MeanProfileWorkspace(alloc_ms)
        scratch, projected = zeros(2alloc_k), zeros(2alloc_k)
        borrowed_identity = v -> copyto!(scratch, v)
        bg.mean_profile_correction(alloc_ms, 0.01, projected, borrowed_identity, alloc_ws)
        kernel_bytes = @allocated bg.mean_profile_correction(
            alloc_ms, 0.01, projected, borrowed_identity, alloc_ws)
        @test kernel_bytes < sizeof(Float64)*(2alloc_k)*alloc_p
    end

    @testset "original-unit fitted means, ownership, and serialization" begin
        original_y = 3.0 .+ 2.0 .* y
        for standardize in (false, true), storage in (:dense, :sparse)
            X = storage == :sparse ? sparse(designs.mixed) : designs.mixed
            ss = suffstats(BipartiteNormalizedModel, f, w, original_y; X=X, standardize=standardize)
            result = fit_mle(BipartiteNormalizedModel, ss;
                solver=ExactCholesky(optim_iters=8, polish=false), seed=71)
            V, yo, Xo, _ = sparse_mean_rows(f, w, original_y, X)
            ref = sparse_mean_oracle(result.model, V, yo, Xo, Matrix{Float64}(I, K, K),
                result.rho, result.sigma_a, result.sigma_z, result.sigma_epsilon)
            @test ss.y_mean .+ Xo * result.beta ≈ Xo * ref.beta atol=1e-8 rtol=1e-8
            @test (result.stats.mean_stats.VtX isa SparseMatrixCSC) == (storage == :sparse)
            @test dof(result) == 4 + size(X, 2)
            @test coefnames(result)[end-2:end] == ["beta_1", "beta_2", "beta_3"]
            buffer = IOBuffer()
            serialize(buffer, (; stats=ss, result))
            seekstart(buffer)
            saved = deserialize(buffer)
            @test (saved.stats.mean_stats.VtX isa SparseMatrixCSC) == (storage == :sparse)
            @test saved.result.beta == result.beta
            @test saved.result.nll == result.nll
            beta_before = copy(saved.result.beta)
            products_before = copy(saved.stats.mean_stats.VtX)
            _ = fit_mle(BipartiteNormalizedModel, saved.stats;
                solver=ExactCholesky(optim_iters=3, polish=false), seed=72)
            @test saved.result.beta == beta_before
            @test saved.stats.mean_stats.VtX == products_before
            @test result.beta == beta_before
        end
    end

    @testset "known mean-weighting defects remain separate" begin
        for storage in (:dense, :sparse)
            convert_X(A) = storage == :sparse ? sparse(A) : A
            # Issue #128: repeated-edge X is taken from the first n_edges
            # input rows, not mapped to the collapsed edges. Controls are
            # constant within each edge, so the intended mapping is unambiguous.
            edge = suffstats(BipartiteNormalizedModel, [1, 1, 2], [1, 1, 2],
                [0.2, 0.4, 0.9]; X=convert_X(reshape([1.0, 1.0, 2.0], 3, 1)),
                weighting=Weighting(observations=:edge), standardize=false)
            @test_broken edge.mean_stats.Xty ≈ [2.1]

            # Issue #129: order the first two rows by unique edge to avoid
            # #128. Observation products update at the candidate rho_eps,
            # but the mean products still use preparation's rho_eps=0.5.
            effective = suffstats(BipartiteNormalizedModel, [1, 2, 1, 2, 2],
                [1, 2, 1, 2, 2], [0.3, 0.8, 0.3, 0.8, 0.8];
                X=convert_X(reshape([1.0, 2.0, 1.0, 2.0, 2.0], 5, 1)),
                weighting=Weighting(observations=:effective, rho_eps=:estimate),
                standardize=false)
            model = BipartiteNormalizedModel(effective.A_prior; rho_limit=0.8)
            theta = [atanh(0.2/0.8), log(0.8), log(0.6), log(0.5),
                     bg.rhoeps_to_unconstrained(0.2)]
            obs = bg.objective_stats(model, effective, theta)
            weights = [2.0, 3.0] ./ (1 .+ ([2.0, 3.0] .- 1) .* 0.2)
            expected = [dot([1.0, 2.0] .* weights, [0.3, 0.8])]
            @test_broken obs.mean_stats.Xty ≈ expected
        end
    end
end
