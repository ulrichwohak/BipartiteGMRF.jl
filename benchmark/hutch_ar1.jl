# Opt-in synthetic benchmark; no application data or additional dependencies.
#
# Accuracy, including bounded fits and reoptimized rho profiles:
# OPENBLAS_NUM_THREADS=1 julia --project=. benchmark/hutch_ar1.jl accuracy
# Skip fits while checking fixed-parameter errors: add fit-iters=0.
#
# Scaling, one fresh process per case (peak RSS includes Julia/JIT/preparation):
# /usr/bin/time -l env OPENBLAS_NUM_THREADS=1 julia --project=. \
#   benchmark/hutch_ar1.jl scaling firms=1000 observations=3000 nodes=6000 controls=8
# Linux: replace /usr/bin/time -l with /usr/bin/time -v.
# Add fit-iters=20 for an explicitly iteration-bounded, end-to-end Hutch fit.
#
# This fixture is a connected, low-fill-in FOREST, not a production-graph
# surrogate. No network factorization is used to simulate it or in scaling
# mode. Dense/ExactCholesky references are restricted to accuracy mode.
using BipartiteGMRF, LinearAlgebra, Random, SparseArrays

const BG = BipartiteGMRF
const PLANTED = (rho=0.3, sigma_a=0.8, sigma_z=0.6, sigma_epsilon=0.7, eta=0.35)

function options(args)
    mode = isempty(args) ? "accuracy" : first(args)
    mode in ("accuracy", "scaling") || error("First argument must be accuracy or scaling")
    defaults = mode == "accuracy" ?
        (firms=8, observations=32, nodes=64, controls=3, fit_iters=100) :
        (firms=1000, observations=3000, nodes=6000, controls=8, fit_iters=0)
    opts = Dict{String,String}(
        "firms" => string(defaults.firms), "observations" => string(defaults.observations),
        "nodes" => string(defaults.nodes), "controls" => string(defaults.controls),
        "fit-iters" => string(defaults.fit_iters), "seed" => "20261007",
        "data-seed" => "20261006", "probes" => "128", "steps" => "32",
        "tol" => "1e-10", "cg-maxiter" => "2000", "repetitions" => "3",
        "probe-counts" => "8,32,128", "lanczos-steps" => "6,16,32",
        "solve-tols" => "1e-4,1e-7,1e-10", "profile-rhos" => "-0.3,0.0,0.3",
        "independent-seeds" => "20261008,20261009")
    for arg in Iterators.drop(args, 1)
        pair = split(arg, '='; limit=2)
        length(pair) == 2 || error("Expected key=value; got $arg")
        haskey(opts, pair[1]) || error("Unknown option $(pair[1])")
        opts[pair[1]] = pair[2]
    end
    integer(key) = parse(Int, opts[key])
    realval(key) = parse(Float64, opts[key])
    list(T, key) = parse.(T, split(opts[key], ','))
    c = (; mode, firms=integer("firms"), observations=integer("observations"),
        nodes=integer("nodes"), controls=integer("controls"), fit_iters=integer("fit-iters"),
        seed=integer("seed"), data_seed=integer("data-seed"), probes=integer("probes"),
        steps=integer("steps"), tol=realval("tol"), cg_maxiter=integer("cg-maxiter"),
        repetitions=integer("repetitions"), probe_counts=list(Int, "probe-counts"),
        lanczos_steps=list(Int, "lanczos-steps"), solve_tols=list(Float64, "solve-tols"),
        profile_rhos=list(Float64, "profile-rhos"),
        independent_seeds=list(Int, "independent-seeds"))
    c.firms >= 2 || error("firms must be at least 2")
    c.observations >= c.firms || error("Every firm must have at least one observation")
    c.nodes >= 2c.firms + c.observations - 1 || error(
        "nodes must be at least 2*firms + observations - 1; extras are graph-only leaves")
    0 <= c.controls < c.observations || error("Require 0 <= controls < observations")
    c.fit_iters >= 0 && c.repetitions > 0 || error("Invalid fit/repetition budget")
    all(abs.(c.profile_rhos) .< 0.99) || error("Profile rho values must lie inside (-0.99, 0.99)")
    c.mode == "accuracy" && max(c.nodes, c.observations) > 512 && error(
        "Accuracy mode has a 512-node/observation dense-reference safety limit")
    return c
end

function git_provenance()
    root = dirname(dirname(pathof(BG)))
    try
        return (; sha=readchomp(`git -C $root rev-parse HEAD`),
            dirty=!isempty(readchomp(`git -C $root status --porcelain --untracked-files=no`)))
    catch
        return (; sha="unavailable", dirty=missing)
    end
end

function emit(record)
    println(record)
    flush(stdout)
end

function measured(f, repetitions)
    f() # warm before measuring; compilation is still included in process RSS
    samples = [@timed f() for _ in 1:repetitions]
    best = samples[argmin(getproperty.(samples, :time))]
    return (; seconds=best.time, allocated_bytes=best.bytes,
        all_seconds=getproperty.(samples, :time))
end

function synthetic_fixture(c)
    rng = MersenneTwister(c.data_seed)
    nf, K, n, p = c.firms, c.observations, c.nodes, c.controls
    nw = n - nf
    extra = n - (2nf + K - 1)
    counts = fill(K ÷ nf, nf)
    counts[1:rem(K, nf)] .+= 1
    # A bridge manager joins neighboring firms. Every other manager is a
    # private leaf, so the declared prior graph has exactly n-1 edges.
    a = zeros(nf); z = zeros(nw)
    innovation = sqrt(1 - PLANTED.rho^2)
    for i in 1:nf
        a[i] = i == 1 ? randn(rng) : PLANTED.rho * z[i - 1] + innovation * randn(rng)
        i < nf && (z[i] = PLANTED.rho * a[i] + innovation * randn(rng))
    end
    Xobs = if p == 0
        spzeros(K, 0)
    elseif p == 1
        sparse(1:K, ones(Int, K), ones(K), K, p)
    else
        sparse(vcat(1:K, 1:K), vcat(ones(Int, K), 2 .+ mod.(0:K-1, p-1)),
            vcat(ones(K), randn(rng, K)), K, p)
    end
    beta = sin.(collect(1:p)) ./ 3
    yobs = Xobs * beta
    f, w, matches, ranks, obsrow = Int[], Int[], Int[], Int[], Int[]
    firm_of, rank_of = zeros(Int, K), zeros(Int, K)
    vi, vj, vv = Int[], Int[], Float64[]
    observation = 0
    for i in 1:nf
        residual = PLANTED.sigma_epsilon * randn(rng)
        for rank in 1:counts[i]
            observation += 1
            worker = nf - 1 + observation
            z[worker] = PLANTED.rho * a[i] + innovation * randn(rng)
            members = [worker]
            rank == 1 && i > 1 && push!(members, i - 1)
            rank == counts[i] && i < nf && push!(members, i)
            rank > 1 && (residual = PLANTED.eta * residual +
                PLANTED.sigma_epsilon * sqrt(1 - PLANTED.eta^2) * randn(rng))
            yobs[observation] += PLANTED.sigma_a * a[i] +
                PLANTED.sigma_z * sum(z[members]) / length(members) + residual
            firm_of[observation] = i; rank_of[observation] = rank
            push!(vi, observation); push!(vj, i); push!(vv, 1.0)
            for member in members
                push!(f, i); push!(w, member); push!(matches, observation)
                push!(ranks, rank); push!(obsrow, observation)
                push!(vi, observation); push!(vj, nf + member)
                push!(vv, inv(Float64(length(members))))
            end
        end
    end
    for j in 1:extra
        i = mod1(j, nf); worker = nf - 1 + K + j
        z[worker] = PLANTED.rho * a[i] + innovation * randn(rng)
        push!(f, i); push!(w, worker); push!(matches, K + j)
        push!(ranks, 0); push!(obsrow, 0) # graph-only rows are not AR time steps
    end
    y = [s == 0 ? NaN : yobs[s] for s in obsrow]
    X = p == 0 ? nothing : Xobs[max.(obsrow, 1), :]
    A = sparse(f, w, ones(length(f)), nf, nw)
    @assert nnz(A) == n - 1
    V = sparse(vi, vj, vv, K, n)
    return (; f, w, y, X, matches, ranks, A, V, Xobs, yobs, firm_of, rank_of,
        graph_only_rows=extra, beta)
end

function prepare(data)
    return suffstats(BipartiteVarianceStableModel, data.f, data.w, data.y;
        n_firms=size(data.A, 1), n_workers=size(data.A, 2),
        weighting=Weighting(observations=:raw), standardize=false,
        match_id=data.matches, edge_index=data.ranks, error_eta=:estimate, X=data.X)
end

function theta(model, p)
    return [atanh(p.rho / BG.rho_limit(model)), log(p.sigma_a), log(p.sigma_z),
        log(p.sigma_epsilon), BG.eta_to_unconstrained(p.eta)]
end

# Independent observation-space oracle: assemble Q, V and R from the fixture,
# not from package precision/AR sufficient-statistics helpers. The constants
# follow package NLL convention: omit K*log(2*pi)/2, use profiled ML (not REML).
function dense_reference(data, p)
    nf, nw = size(data.A)
    df = vec(sum(data.A; dims=2)); dw = vec(sum(data.A; dims=1))
    denom = 1 - p.rho^2
    Q = [Diagonal((1 .+ p.rho^2 .* (df .- 1)) ./ (denom * p.sigma_a^2)) -p.rho .* Matrix(data.A) ./ (denom * p.sigma_a * p.sigma_z);
         -p.rho .* Matrix(data.A') ./ (denom * p.sigma_a * p.sigma_z) Diagonal((1 .+ p.rho^2 .* (dw .- 1)) ./ (denom * p.sigma_z^2))]
    K = length(data.yobs)
    R = [data.firm_of[i] == data.firm_of[j] ?
        p.eta^abs(data.rank_of[i] - data.rank_of[j]) : 0.0 for i in 1:K, j in 1:K]
    V = Matrix(data.V)
    Sigma = Symmetric(V * (Symmetric(Q) \ V') + p.sigma_epsilon^2 .* R)
    factor = cholesky(Sigma)
    y = data.yobs
    X = Matrix(data.Xobs)
    beta = size(X, 2) == 0 ? Float64[] : (X' * (factor \ X)) \ (X' * (factor \ y))
    residual = y - X * beta
    return (; nll=0.5 * (logdet(factor) + dot(residual, factor \ residual)), beta)
end

function evaluator(solver, model, stats, seed)
    cache = BG.make_nll_cache(solver, model, stats)
    function evaluate(p)
        t = theta(model, p)
        obs = BG.objective_stats(model, stats, t)
        value = BG.nll_value(solver, model, stats, t, obs, cache; seed)
        isfinite(value) && value < BG.BIG_NLL || error("Invalid objective at $p")
        return value
    end
    return evaluate, cache
end

function hutch(c; probes=c.probes, steps=c.steps, tol=c.tol)
    return HutchSLQ(logdet_probes=probes, lanczos_iters=steps, cg_tol=tol,
        cg_maxiter=c.cg_maxiter, optim_iters=max(c.fit_iters, 1), g_reltol=1e-10,
        simplex_scale=0.0, simplex_shift=0.1)
end

result_parameters(r) = (; rho=r.rho, sigma_a=r.sigma_a, sigma_z=r.sigma_z,
    sigma_epsilon=r.sigma_epsilon, eta=r.eta)

function accuracy(c, data, stats, model)
    exact = ExactCholesky(optim_iters=max(c.fit_iters, 1), polish=false,
        g_reltol=1e-10, simplex_scale=0.0, simplex_shift=0.1)
    exact_value, _ = evaluator(exact, model, stats, c.seed)
    # Each axis changes independently, holding the other two at their baseline.
    settings = unique(vcat(
        [(axis="probes", probes=m, steps=c.steps, tol=c.tol) for m in c.probe_counts],
        [(axis="lanczos", probes=c.probes, steps=k, tol=c.tol) for k in c.lanczos_steps],
        [(axis="solve", probes=c.probes, steps=c.steps, tol=t) for t in c.solve_tols]))
    for setting in settings
        value, _ = evaluator(hutch(c; probes=setting.probes, steps=setting.steps,
            tol=setting.tol), model, stats, c.seed)
        # Revisit -0.45 after crossing zero, without rebuilding the cache.
        for eta in (-0.45, 0.0, 0.45, -0.45), rho in (-0.3, 0.0, 0.3)
            p = merge(PLANTED, (; eta, rho))
            reference = dense_reference(data, p).nll
            exact_nll = exact_value(p)
            isapprox(exact_nll, reference; atol=1e-8, rtol=1e-9) ||
                error("Exact/dense reference disagreement")
            estimate = value(p)
            emit((; stage="accuracy", setting..., seed=c.seed, p..., dense_nll=reference,
                exact_nll, hutch_nll=estimate, hutch_minus_exact=estimate-exact_nll,
                error_per_observation=(estimate-exact_nll)/stats.K))
        end
    end
    for seed in c.independent_seeds
        value, _ = evaluator(hutch(c), model, stats, seed)
        estimate = value(PLANTED); reference = exact_value(PLANTED)
        emit((; stage="independent_seed_accuracy", seed, probes=c.probes,
            steps=c.steps, tol=c.tol, exact_nll=reference, hutch_nll=estimate,
            hutch_minus_exact=estimate-reference))
    end
    c.fit_iters == 0 && return
    starts = [PLANTED, merge(PLANTED, (; rho=-0.2, eta=-0.2,
        sigma_a=0.6, sigma_z=0.8, sigma_epsilon=0.9))]
    # Establish exact reference candidates for both starts and each profile
    # point. All optimizations have the same explicit iteration budget.
    fit_cases = vcat([(; fix_rho=nothing, start=i) for i in eachindex(starts)],
        [(; fix_rho=rho, start=1) for rho in c.profile_rhos])
    exact_results = Dict{Tuple{Union{Nothing,Float64},Int},Any}()
    for case in fit_cases
        init = starts[case.start]
        case.fix_rho === nothing || (init=merge(init, (; rho=case.fix_rho)))
        r = fit_mle(model, stats; solver=exact, init, fix_rho=case.fix_rho, seed=c.seed)
        exact_results[(case.fix_rho, case.start)] = r
        oracle = dense_reference(data, result_parameters(r))
        isapprox(r.nll, oracle.nll; atol=1e-7, rtol=1e-8) || error("Fit/dense mismatch")
        emit((; stage="exact_fit", case..., seed=c.seed, budget=c.fit_iters,
            result_parameters(r)..., nll=r.nll, converged=r.converged,
            iterations=r.iterations, obj_evals=r.obj_evals))
    end
    # Same seed across settings, starts, and reoptimized profile comparisons.
    # Independent-seed fits below assess sensitivity separately.
    fit_settings = vcat(
        [(; probes=m, seed=c.seed) for m in c.probe_counts],
        [(; probes=c.probes, seed=s) for s in c.independent_seeds])
    for setting in fit_settings, case in fit_cases
        init = starts[case.start]
        case.fix_rho === nothing || (init=merge(init, (; rho=case.fix_rho)))
        elapsed = @timed fit_mle(model, stats; solver=hutch(c; probes=setting.probes),
            init, fix_rho=case.fix_rho, seed=setting.seed)
        r = elapsed.value
        p = result_parameters(r)
        ref = exact_results[(case.fix_rho, case.start)]
        exact_at_candidate = exact_value(p)
        emit((; stage=case.fix_rho === nothing ? "hutch_fit" : "rho_profile_fit",
            case..., setting..., steps=c.steps, tol=c.tol, budget=c.fit_iters, p...,
            nll=r.nll, exact_at_candidate,
            approximation_error=r.nll-exact_at_candidate,
            exact_objective_gap=exact_at_candidate-ref.nll,
            parameter_delta=map(-, values(p), values(result_parameters(ref))),
            converged=r.converged, iterations=r.iterations, obj_evals=r.obj_evals,
            seconds=elapsed.time, allocated_bytes=elapsed.bytes))
    end
end

function scaling(c, stats, model)
    solver = hutch(c)
    value, cache = evaluator(solver, model, stats, c.seed)
    emit((; stage="fixed_evaluation", seed=c.seed, nll=value(PLANTED),
        measurement=measured(() -> value(PLANTED), c.repetitions),
        stats_bytes=Base.summarysize(stats), cache_bytes=Base.summarysize(cache),
        mean_product_type=stats.mean_stats === nothing ? "none" : string(typeof(stats.mean_stats.VtX))))
    # Explicitly include eta-changing reconstruction/update cost in timing.
    etas = [-0.35, 0.0, 0.35]
    cycle() = [value(merge(PLANTED, (; eta))) for eta in etas]
    emit((; stage="eta_cycle", eta_values=etas,
        measurement=measured(cycle, c.repetitions), evaluations_per_cycle=length(etas)))
    c.fit_iters == 0 && return
    fit = @timed fit_mle(model, stats; solver, init=PLANTED, seed=c.seed)
    r = fit.value
    emit((; stage="bounded_end_to_end_fit", seed=c.seed, budget=c.fit_iters,
        result_parameters(r)..., nll=r.nll, converged=r.converged,
        iterations=r.iterations, obj_evals=r.obj_evals,
        seconds=fit.time, allocated_bytes=fit.bytes))
end

function main(args)
    c = options(args)
    emit((; stage="provenance", julia=VERSION, package_version=pkgversion(BG),
        source=pathof(BG), git=git_provenance(), cpu=Sys.cpu_info()[1].model,
        ram_bytes=Sys.total_memory(), julia_threads=Threads.nthreads(),
        blas_threads=BLAS.get_num_threads(), config=c,
        likelihood_constants="omits K*log(2*pi)/2; profiled ML, not REML",
        graph="connected bipartite forest; graph-only leaf nodes retained",
        caveat="Synthetic, low-fill graph; not a production memory extrapolation. RSS is external."))
    data = synthetic_fixture(c)
    prep = @timed prepare(data)
    stats = prep.value
    model = BipartiteVarianceStableModel(stats.A_prior; strict_forest=true, rho_limit=0.99)
    @assert stats.K == c.observations
    @assert stats.N_firms + stats.N_workers == c.nodes
    @assert stats.metadata.graph_only_rows == data.graph_only_rows
    @assert stats.A_prior == data.A
    emit((; stage="fixture", input_rows=length(data.y), grouped_observations=stats.K,
        nodes=c.nodes, graph_edges=nnz(data.A), graph_only_rows=data.graph_only_rows,
        controls=c.controls, rho_limit=BG.rho_limit(model),
        preparation_seconds=prep.time, preparation_allocated_bytes=prep.bytes,
        stats_bytes=Base.summarysize(stats)))
    c.mode == "accuracy" ? accuracy(c, data, stats, model) : scaling(c, stats, model)
    emit((; stage="complete", mode=c.mode,
        interpretation="Iteration-limited fits are candidates, not recovery/convergence guarantees."))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
