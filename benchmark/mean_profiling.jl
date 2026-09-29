# Opt-in, synthetic-only mean-profiling benchmark (not part of Pkg.test()).
# Usage: julia --project=. benchmark/mean_profiling.jl [firms=1000] [p=128]
#        [components=1] [iid|ar1] [sparse|dense]
# Run each case in a fresh process under /usr/bin/time -l (macOS) or
# /usr/bin/time -v (Linux) to measure process peak RSS, including Julia/JIT.
# This script also runs against v0.5.3 for identical-fixture comparisons.
using BipartiteGMRF, LinearAlgebra, Random, SparseArrays

const BG = BipartiteGMRF
const GM = BG.GaussianMarkovRandomFields
const SEED = 20260929

function fixture(m, p, components, storage)
    rng = MersenneTwister(SEED)
    span = m ÷ components
    f, w, edge_index = Int[], Int[], Int[]
    for j in 1:m
        next_worker = ((j - 1) ÷ span) * span + mod(j, span) + 1
        append!(f, (j, j))
        append!(w, (j, next_worker))
        append!(edge_index, (1, 2))
    end
    K = length(f)
    columns = mod.(collect(0:K-1), p) .+ 1
    # Two entries per row: an indicator and a signed fractional exposure.
    # This is not a categorical-only design; every column has support.
    X = sparse(repeat(1:K, 2), vcat(columns, mod.(columns, p) .+ 1),
        vcat(ones(K), 0.2 .* randn(rng, K)), K, p)
    y = X * (sin.(collect(1:p)) ./ 5) + randn(rng, K)
    return (; f, w, y, X=storage == "dense" ? Matrix(X) : X, edge_index)
end

function measured(f; repetitions=3)
    f()  # compile and warm this operation before recording it
    samples = [(@timed f()) for _ in 1:repetitions]
    sample = samples[argmin(getproperty.(samples, :time))]
    return (; seconds=sample.time, allocated_bytes=sample.bytes)
end

# A bounded-memory reference for isolating the dense p-by-p solve cost.
# Never retain all n-by-p solved columns in the benchmark itself.
function coefficient_system(ms, lambda, projected_y, solve_M)
    c = lambda .* ms.Xty - lambda^2 .* (ms.VtX' * solve_M(projected_y))
    G = lambda .* ms.XtX
    for j in 1:ms.p
        G[:, j] .-= lambda^2 .* (ms.VtX' * solve_M(Vector(ms.VtX[:, j])))
    end
    return Symmetric(G), c
end

function git_provenance()
    root = dirname(dirname(pathof(BG)))
    try
        sha = readchomp(`git -C $root rev-parse HEAD`)
        dirty = !isempty(readchomp(`git -C $root status --porcelain --untracked-files=no`))
        return (; sha, dirty)
    catch
        return (; sha="unavailable", dirty=missing)
    end
end

function main(args)
    length(args) <= 5 || error("Expected: firms p components iid|ar1 sparse|dense")
    m = length(args) >= 1 ? parse(Int, args[1]) : 1000
    p = length(args) >= 2 ? parse(Int, args[2]) : 128
    components = length(args) >= 3 ? parse(Int, args[3]) : 1
    error_kind = length(args) >= 4 ? args[4] : "iid"
    storage = length(args) >= 5 ? args[5] : "sparse"
    m >= 2 && 1 <= components <= m ÷ 2 && m % components == 0 ||
        error("Use firms >= 2 and a component count dividing firms, with >= 2 firms/component")
    2 <= p < 2m || error("Use 2 <= p < the observation count (2*firms)")
    error_kind in ("iid", "ar1") || error("Error kind must be iid or ar1")
    storage in ("sparse", "dense") || error("Storage must be sparse or dense")
    # Conservative array-budget gate for BOTH revisions. This is not an RSS
    # bound: Julia/JIT, native factors, and allocator high-water marks add cost.
    dense_array_budget = big(8) * (8 * big(2m) * p + 12 * big(p)^2)
    dense_array_budget <= 512 * 1024^2 ||
        error("Synthetic case exceeds the 512 MiB dense-array preflight budget; use a smaller case")
    println((; stage="provenance", julia=VERSION, package_version=pkgversion(BG),
        gmrf_version=pkgversion(GM), source=pathof(BG), git=git_provenance(),
        cpu=Sys.cpu_info()[1].model, ram_bytes=Sys.total_memory(),
        julia_threads=Threads.nthreads(), blas_threads=BLAS.get_num_threads(),
        seed=SEED, n=2m, K=2m, p, components, error_kind, storage,
        dense_array_budget=Int(dense_array_budget)))
    flush(stdout)

    data = fixture(m, p, components, storage)
    kwargs = error_kind == "ar1" ?
        (; error_eta=:estimate, edge_index=data.edge_index) : (;)
    prepare() = suffstats(BipartiteNormalizedModel, data.f, data.w, data.y;
        X=data.X, weighting=Weighting(observations=:raw), standardize=false, kwargs...)
    prep_measure = measured(prepare)
    stats = prepare()
    model = BipartiteNormalizedModel(stats.A_prior)
    theta = [atanh(0.2 / BG.rho_limit(model)), log(0.8), log(0.6), log(0.7)]
    error_kind == "ar1" && push!(theta, BG.eta_to_unconstrained(0.2))
    obs = BG.objective_stats(model, stats, theta)
    cache = BG.make_nll_cache(ExactCholesky(), model, stats)
    objective() = BG.nll_exact_value(model, stats, theta,
        BG.objective_stats(model, stats, theta), cache)
    nll = objective()
    isfinite(nll) && nll < BG.BIG_NLL || error("Invalid fixed-parameter objective")
    lambda = 1 / 0.7^2
    solve_M = v -> GM.workspace_solve(cache.ws_M, v)
    ms = obs.mean_stats
    profile_workspace = isdefined(BG, :MeanProfileWorkspace) ? BG.MeanProfileWorkspace(ms) : nothing
    function profile()
        if profile_workspace === nothing
            return BG.mean_profile_correction(ms, lambda, obs.design.projected_y, solve_M)
        end
        return BG.mean_profile_correction(ms, lambda, obs.design.projected_y, solve_M,
            profile_workspace)
    end
    correction, beta = profile()
    rhs = Vector(ms.VtX[:, 1])
    block_size = profile_workspace === nothing ? p : size(profile_workspace.solved, 2)
    product_width = min(8, p) # same bounded product instrumentation on both revisions
    solved_block = hcat((solve_M(Vector(ms.VtX[:, j])) for j in 1:product_width)...)
    G, c = coefficient_system(ms, lambda, obs.design.projected_y, solve_M)
    precision_values = copy(nonzeros(cache.ws_M.Q))
    function refactor()
        GM.update_precision_values!(cache.ws_M, precision_values)
        GM.ensure_numeric!(cache.ws_M)
    end
    decoded = BG.unpack_params(theta; rho_limit=BG.rho_limit(model))
    function reconstruction()
        if isdefined(BG, :final_mean_profile)
            return BG.final_mean_profile(ExactCholesky(), model, stats, obs, decoded, cache)
        end
        Q = BG.model_precision(model, decoded.rho, decoded.sigma_a, decoded.sigma_z)
        ws = GM.GMRFWorkspace(Q + lambda .* obs.design.VtV)
        GM.ensure_numeric!(ws)
        return BG.mean_profile_correction(ms, lambda, obs.design.projected_y,
            v -> GM.workspace_solve(ws, v))
    end
    objective_measure = measured(objective)
    reconstruction_measure = measured(reconstruction)
    fit() = fit_mle(BipartiteNormalizedModel, stats;
        solver=ExactCholesky(optim_iters=5, polish=false), fix_rho=0.2, seed=SEED)
    fit()  # warm end-to-end fit separately
    pilot = @timed fit()
    result = pilot.value
    # This gate uses this pilot's actual objective count, not an unrelated fit.
    # It excludes optimizer/cache-setup overhead; compare it with measured pilot.
    projected_core_seconds = prep_measure.seconds +
        result.obj_evals * objective_measure.seconds + reconstruction_measure.seconds
    numerical_nnz = issparse(ms.VtX) ? count(!iszero, nonzeros(ms.VtX)) : count(!iszero, ms.VtX)
    println((; stage="measurements", VtX_type=string(typeof(ms.VtX)),
        nnz_X=issparse(data.X) ? nnz(data.X) : count(!iszero, data.X),
        VtX_stored_entries=issparse(ms.VtX) ? nnz(ms.VtX) : length(ms.VtX),
        VtX_density=numerical_nnz / length(ms.VtX), block_size, product_width,
        stats_bytes=Base.summarysize(stats), VtX_bytes=Base.summarysize(ms.VtX),
        profile_workspace_bytes=Base.summarysize(profile_workspace),
        prep=prep_measure,
        observation_combination=measured(() -> BG.objective_stats(model, stats, theta)),
        solve=measured(() -> solve_M(rhs)), numeric_factorization=measured(refactor),
        block_product=measured(() -> ms.VtX' * solved_block),
        coefficient_solve=measured(() -> cholesky(G) \ c),
        profile=measured(profile), objective=objective_measure, reconstruction=reconstruction_measure,
        pilot_seconds=pilot.time, pilot_allocated_bytes=pilot.bytes,
        pilot_obj_evals=result.obj_evals, pilot_converged=result.converged,
        projected_core_seconds, nll, correction, beta_norm=norm(beta),
        pilot_nll=result.nll, pilot_beta_norm=norm(result.beta)))
    # Returning these small vectors also permits full old/new coefficient
    # comparisons from a driver script without printing hundreds of values.
    return (; beta, pilot_beta=result.beta, nll, pilot_nll=result.nll)
end

main(ARGS)
