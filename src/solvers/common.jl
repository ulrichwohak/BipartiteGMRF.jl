function validate_capability(model::AbstractBipartiteModel, stats::BipartiteGMRFStats, solver::AbstractGMRFSolver)
    if model isa BipartiteVarianceStableModel
        stats.weighting.observations == :raw ||
            throw(ArgumentError("BipartiteVarianceStableModel currently supports only raw observation weighting."))
    end
    if model isa BipartiteSpectralModel && solver isa ExactCholesky
        throw(ArgumentError("ExactCholesky for BipartiteSpectralModel is not implemented; use HutchSLQ."))
    end
    if stats.error_classes !== nothing && length(stats.error_classes.counts) > 1 &&
       !(solver isa ExactCholesky)
        throw(ArgumentError("error_groups currently supports only the ExactCholesky solver."))
    end
    return nothing
end

# ─── Precision matrix via model dispatch ──────────────────────────────────

"""
Build the model precision matrix Q at given parameters.
Delegates to the `LatentModel` interface.
"""
function model_precision(model::AbstractBipartiteModel, rho::Float64, sigma_a::Float64, sigma_z::Float64)
    return GaussianMarkovRandomFields.precision_matrix(model; ρ=rho, σ_a=sigma_a, σ_z=sigma_z)
end

"""
Build the data-augmented precision M = Q + λV'V.
"""
function fitted_precision(model::AbstractBipartiteModel, VtV::SparseMatrixCSC, rho::Float64, sigma_a::Float64, sigma_z::Float64, sigma_epsilon::Float64)
    Q = model_precision(model, rho, sigma_a, sigma_z)
    return Q + (1.0 / sigma_epsilon^2) .* VtV
end

# ─── Matrix-free operators (HutchSLQ path) ────────────────────────────────

q_operator(model::BipartiteVarianceStableModel, rho::Float64, sigma_a::Float64, sigma_z::Float64) =
    make_qop_vs(model, rho, sigma_a, sigma_z)
q_operator(model::AbstractBipartiteModel, rho::Float64, sigma_a::Float64, sigma_z::Float64) =
    make_qop(model, rho, sigma_a, sigma_z)

# Q has a constant diagonal for the normalized and spectral models.
function q_diag(
    model::Union{BipartiteNormalizedModel,BipartiteSpectralModel},
    rho::Float64, sigma_a::Float64, sigma_z::Float64,
)
    g = model.graph
    return vcat(
        fill(1.0 / sigma_a^2, g.n_firms),
        fill(1.0 / sigma_z^2, g.n_workers),
    )
end

function q_diag(model::BipartiteUnnormalizedModel, rho::Float64, sigma_a::Float64, sigma_z::Float64)
    g = model.graph
    return vcat(g.d_f ./ sigma_a^2, g.d_w ./ sigma_z^2)
end

function q_diag(model::BipartiteVarianceStableModel, rho::Float64, sigma_a::Float64, sigma_z::Float64)
    g = model.graph
    rho_sq = rho^2
    inv_one_minus_rho_sq = 1.0 / (1.0 - rho_sq)
    return vcat(
        (1.0 .+ rho_sq .* (g.d_f .- 1.0)) .* (inv_one_minus_rho_sq / sigma_a^2),
        (1.0 .+ rho_sq .* (g.d_w .- 1.0)) .* (inv_one_minus_rho_sq / sigma_z^2),
    )
end

# ─── Observation stats ─────────────────────────────────────────────────────

function objective_stats(model::AbstractBipartiteModel, stats::BipartiteGMRFStats, params_full::Vector{Float64})
    w = stats.weighting
    if w.observations == :effective && w.estimate_rho_eps
        rho_eps = rhoeps_from_unconstrained(params_full[5])
        design, weight_stats = build_match_weight_stats(
            stats.base.f,
            stats.base.w,
            stats.base.y,
            stats.base.T,
            model.graph.n_firms,
            model.graph.n_workers,
            rho_eps,
        )
        return ObservationStats(design, weight_stats, rho_eps, stats.mean_stats)
    end
    ec = stats.error_classes
    if ec !== nothing && length(ec.counts) > 1
        # Group-robust errors: params 5.. are log omega for classes 2..C
        # (class 1, the smallest group size, is pinned at omega = 1). The
        # class blocks are combined with weights 1/omega_c into V'ΛV, V'Λy,
        # y'Λy on the pooled sparsity pattern; the likelihood constant
        # Σ_c K_c log omega_c rides in -log_weight_sum.
        C = length(ec.counts)
        winv = ones(Float64, C)
        lws = 0.0
        for c in 2:C
            lw = params_full[3 + c]
            winv[c] = exp(-lw)
            lws -= ec.counts[c] * lw
        end
        P = stats.design.VtV
        n = size(P, 1)
        nf = model.graph.n_firms
        VtV = SparseMatrixCSC(n, n, P.colptr, P.rowval, ec.vtv_nzvals * winv)
        FF = SparseMatrixCSC{Float64,Int}(VtV[1:nf, 1:nf])
        WW = SparseMatrixCSC{Float64,Int}(VtV[(nf+1):n, (nf+1):n])
        A_obs = SparseMatrixCSC{Float64,Int}(VtV[1:nf, (nf+1):n])
        design = DesignStats(VtV, ec.projected * winv, dot(ec.ydot, winv), A_obs,
                             SparseMatrixCSC{Float64,Int}(copy(transpose(A_obs))), FF, WW)
        ms = ec.mean_stats === nothing ? nothing :
            MeanStats(sum(winv[c] .* ec.mean_stats[c].VtX for c in 1:C),
                      sum(winv[c] .* ec.mean_stats[c].XtX for c in 1:C),
                      sum(winv[c] .* ec.mean_stats[c].Xty for c in 1:C),
                      ec.mean_stats[1].p)
        weights = WeightStats(lws, Float64(stats.K), Float64(stats.K), 1.0, 1.0)
        return ObservationStats(design, weights, stats.rho_eps_likelihood, ms)
    end
    return ObservationStats(stats.design, stats.weights, stats.rho_eps_likelihood, stats.mean_stats)
end

function residual_corr_term(stats::BipartiteGMRFStats, sigma_epsilon::Float64, rho_eps::Union{Nothing,Float64})
    rho_eps === nothing && return 0.0
    0.0 <= rho_eps < 1.0 || return BIG_NLL
    omr = 1.0 - rho_eps
    lambda = 1.0 / sigma_epsilon^2
    return Float64(stats.within_df) * (2.0 * log(sigma_epsilon) + log(omr)) +
        lambda * Float64(stats.within_ss) / omr
end

# ═══════════════════════════════════════════════════════════════════════════
# Optimization loop
# ═══════════════════════════════════════════════════════════════════════════
#
# Solver-specific behavior enters through four methods, each dispatching on
# the solver type (implemented in exact.jl and hutch.jl):
#
#   make_nll_cache(solver, model, stats)             -> cache
#   nll_value(solver, model, stats, p, obs, cache; seed) -> Float64
#   nelder_g_abstol(solver, g_rel)                   -> Float64
#   polish(solver, obj, res, verbose)                -> (res, elapsed_seconds)

"""
Compute the profiled β correction term and β̂ from mean-structure statistics.
Returns `(correction, beta)` where `correction` is the scalar to subtract
from the NLL and `beta` is the profiled-out coefficient vector.
Returns `(0.0, nothing)` when there is no mean structure.
"""
function mean_profile_correction(
    ms::MeanStats,
    lambda::Float64,
    projected_y::Vector{Float64},
    solve_M::Function,   # v -> M^{-1}v
)
    # M^{-1} V'y (reuse if already computed, but solve_M is cheap here)
    Minv_Vy = solve_M(projected_y)
    # M^{-1} V'X
    Minv_VtX = similar(ms.VtX)
    for j in 1:ms.p
        Minv_VtX[:, j] = solve_M(ms.VtX[:, j])
    end
    # c = X'Ω^{-1}y = λX'y - λ²(V'X)'M^{-1}V'y
    c = lambda .* ms.Xty .- lambda^2 .* (ms.VtX' * Minv_Vy)
    # G = X'Ω^{-1}X = λX'X - λ²(V'X)'M^{-1}V'X
    G = Symmetric(lambda .* ms.XtX .- lambda^2 .* (ms.VtX' * Minv_VtX))
    G_chol = cholesky(G)
    beta = G_chol \ c
    correction = dot(c, beta)
    return correction, beta
end

function build_gmrf_result(fit, solver::AbstractGMRFSolver, fix_rho::Union{Nothing,Float64})
    return GMRFResult(
        fit.rho,
        fit.sigma_a,
        fit.sigma_z,
        fit.sigma_epsilon,
        fit.rho_eps,
        fit.beta,
        fit.nll,
        fit.converged,
        fit.iterations,
        fit.obj_evals,
        fit.optimization_time,
        fit.model,
        fit.stats,
        solver,
        fit.theta_unconstrained,
        fit.omega === nothing ?
            fit_result_metadata(fit.model, fit.rho, fix_rho) :
            merge(fit_result_metadata(fit.model, fit.rho, fix_rho),
                  (error_class_sizes = vcat(fit.omega_sizes),
                   error_class_variances = vcat(1.0, fit.omega))),
    )
end

function optimize_problem(
    model::AbstractBipartiteModel,
    stats::BipartiteGMRFStats,
    solver::AbstractGMRFSolver;
    fix_rho::Union{Nothing,Float64}=nothing,
    seed::Int=42,
    verbose::Bool=false,
)
    validate_capability(model, stats, solver)
    limit = rho_limit(model)
    if fix_rho !== nothing && !(abs(fix_rho) < limit)
        throw(ArgumentError("fix_rho must lie in (-$(limit), $(limit))."))
    end

    estimate_rho_eps = stats.weighting.observations == :effective &&
        stats.weighting.estimate_rho_eps
    p0 = initial_params(fix_rho, estimate_rho_eps; rho_limit=limit)
    n_omega = stats.error_classes === nothing ? 0 : length(stats.error_classes.counts) - 1
    n_omega > 0 && append!(p0, zeros(n_omega))
    evals = Ref(0)
    cache = make_nll_cache(solver, model, stats)

    function obj(pfree)
        evals[] += 1
        pfull = full_params(Vector{Float64}(pfree), fix_rho, estimate_rho_eps; rho_limit=limit)
        obs = objective_stats(model, stats, pfull)
        return nll_value(solver, model, stats, pfull, obs, cache; seed=seed)
    end

    f0 = obj(p0)
    fscale = (isfinite(f0) && f0 < BIG_NLL) ? max(1.0, abs(f0)) : 1.0
    g_abstol = nelder_g_abstol(solver, solver.g_reltol * fscale)
    opts = Options(iterations=solver.optim_iters, show_trace=verbose, g_tol=g_abstol)
    elapsed = @elapsed res = optimize(obj, p0, NelderMead(), opts)
    res, polish_elapsed = polish(solver, obj, res, verbose)
    elapsed += polish_elapsed

    pfree = Vector{Float64}(minimizer(res))
    pfull = full_params(pfree, fix_rho, estimate_rho_eps; rho_limit=limit)
    obs = objective_stats(model, stats, pfull)
    final_stats = if estimate_rho_eps
        replace_stats(stats;
            design=obs.design, weights=obs.weights, rho_eps_likelihood=obs.rho_eps)
    elseif n_omega > 0
        # Store the omega-weighted design so decompositions and posterior
        # objects built from the result use the fitted error weighting.
        replace_stats(stats; design=obs.design, weights=obs.weights,
            mean_stats=obs.mean_stats)
    else
        stats
    end
    val = obj(pfree)
    decoded = unpack_params(pfull; rho_limit=limit)
    rho_eps = estimate_rho_eps ? obs.rho_eps : stats.rho_eps_likelihood

    # Compute profiled beta at final parameters
    beta_original = if final_stats.mean_stats !== nothing
        lambda_final = 1.0 / decoded.sigma_epsilon^2
        Q_final = model_precision(model, decoded.rho, decoded.sigma_a, decoded.sigma_z)
        M_final = Q_final + lambda_final .* obs.design.VtV
        ws_M = GaussianMarkovRandomFields.GMRFWorkspace(M_final)
        GaussianMarkovRandomFields.ensure_numeric!(ws_M)
        solve_M = v -> GaussianMarkovRandomFields.workspace_solve(ws_M, v)
        _, beta_std = mean_profile_correction(final_stats.mean_stats, lambda_final,
            obs.design.projected_y, solve_M)
        beta_std .* final_stats.y_std
    else
        nothing
    end

    return (
        rho = decoded.rho,
        sigma_a = decoded.sigma_a * final_stats.y_std,
        sigma_z = decoded.sigma_z * final_stats.y_std,
        sigma_epsilon = decoded.sigma_epsilon * final_stats.y_std,
        rho_eps = rho_eps,
        beta = beta_original,
        nll = val,
        converged = converged(res),
        iterations = optim_iterations(res),
        obj_evals = evals[],
        optimization_time = elapsed,
        theta_unconstrained = pfull,
        model = model,
        stats = final_stats,
        omega = n_omega > 0 ? exp.(pfull[5:4+n_omega]) : nothing,
        omega_sizes = n_omega > 0 ? stats.error_classes.sizes : nothing,
    )
end

"""
    solve(model::AbstractBipartiteModel, stats::BipartiteGMRFStats, solver::AbstractGMRFSolver;
          fix_rho=nothing, seed=42, verbose=false)

Fit a bipartite GMRF by maximum likelihood with the given solver. Extends
`CommonSolve.solve`, so it composes with `using LinearSolve` or other
SciML-style packages without name clashes. [`fit_mle`](@ref) is the
higher-level entry point and delegates here.

`fix_rho` fixes the local-dependence parameter during optimization. Use
[`decompose`](@ref) on the returned result for variance decompositions.
"""
function solve(
    model::AbstractBipartiteModel,
    stats::BipartiteGMRFStats,
    solver::AbstractGMRFSolver;
    fix_rho::Union{Nothing,Float64}=nothing,
    seed::Int=42,
    verbose::Bool=false,
)
    fit = optimize_problem(model, stats, solver; fix_rho=fix_rho, seed=seed, verbose=verbose)
    return build_gmrf_result(fit, solver, fix_rho)
end
