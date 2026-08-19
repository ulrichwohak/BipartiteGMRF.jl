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
    if stats.error_ar1 !== nothing && !(solver isa ExactCholesky)
        throw(ArgumentError("error_eta currently supports only the ExactCholesky solver."))
    end
    if stats.error_blocks !== nothing
        solver isa EMIWBlocks ||
            throw(ArgumentError("error_blocks=:iw requires the EMIWBlocks solver."))
        model isa BipartiteVarianceStableModel ||
            throw(ArgumentError("error_blocks=:iw requires the BipartiteVarianceStableModel."))
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
    ar = stats.error_ar1
    if ar !== nothing
        # error_eta is mutually exclusive with rho_eps (observations mode) and
        # error_groups, so the eta codec is the 5th entry of params_full.
        eta = ar.eta_fixed === nothing ? eta_from_unconstrained(params_full[5]) : ar.eta_fixed
        design, weights, ms = ar1_observation_stats(ar, eta, model.graph.n_firms)
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
# Solver-specific behavior enters through five methods, each dispatching on
# the solver type (implemented in exact.jl and hutch.jl):
#
#   make_nll_cache(solver, model, stats)             -> cache
#   nll_value(solver, model, stats, p, obs, cache; seed) -> Float64
#   nelder_g_abstol(solver, g_rel)                   -> Float64
#   nelder_simplexer(solver)                         -> Optim.Simplexer
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
    meta = fit_result_metadata(fit.model, fit.rho, fix_rho)
    if fit.omega !== nothing
        meta = merge(meta, (error_class_sizes = vcat(fit.omega_sizes),
                            error_class_variances = vcat(1.0, fit.omega)))
    end
    fit.eta !== nothing && (meta = merge(meta, (error_eta = fit.eta,)))
    return GMRFResult(
        fit.rho,
        fit.sigma_a,
        fit.sigma_z,
        fit.sigma_epsilon,
        fit.rho_eps,
        fit.eta,
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
        meta,
    )
end

function build_emiw_result(fit, solver::EMIWBlocks, fix_rho::Union{Nothing,Float64})
    base = fit_result_metadata(fit.model, fit.rho, fix_rho)
    return GMRFResult(
        fit.rho,
        fit.sigma_a,
        fit.sigma_z,
        fit.sigma_epsilon,
        fit.rho_eps,
        nothing,  # eta (not part of the integrated-likelihood model)
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
        merge(base, (
            # integrated-likelihood blocks: Ω_i marginalized out under
            # IW(δ + m_i − 1, ∝ φR(r)); nll is −ELBO (a bound, not the exact
            # marginal NLL — see EMIWBlocks docs).
            omega_bar = fit.omega_bar,          # mean error variance = sigma_epsilon²
            error_scale_phi = fit.phi,
            error_corr_r = fit.r,
            t_dof_delta = fit.delta,
            u_bar = fit.u_bar,
            elbo = -fit.nll,
            objective = :elbo,
            block_sizes = fit.stats.error_blocks.sizes,
            sigma_a_network_identified = true,
            unidentified_at_rho_zero = abs(fit.rho) < 1e-8,
        )),
    )
end

"""
    init_status(stats, fix_rho, estimate_rho_eps, n_omega, estimate_eta)

Which optional parameters this fit estimates (`:free`), does not have at all
(`:absent`), or holds fixed — reported as the pinned value itself, so that
[`validate_init`](@ref) can stay quiet when an `init` field merely repeats it.
An `init` field for a parameter the fit never touches is reported rather than
silently dropped.

`rho_eps` is `:absent` under `:raw` and `:edge` weighting, and the ω ladder and
η are `:raw`-only (`suffstats` enforces this), so at most one of `rho_eps`, ω,
η is ever present — which is what lets them share slot 5 of the parameter
vector.
"""
function init_status(
    stats::BipartiteGMRFStats,
    fix_rho::Union{Nothing,Float64},
    estimate_rho_eps::Bool,
    n_omega::Int,
    estimate_eta::Bool,
)
    return (
        rho = fix_rho === nothing ? :free : fix_rho,
        rho_eps = stats.weighting.observations == :effective ?
            (estimate_rho_eps ? :free : stats.weighting.rho_eps) : :absent,
        eta = stats.error_ar1 === nothing ? :absent :
            (estimate_eta ? :free : stats.error_ar1.eta_fixed),
        omega = n_omega > 0 ? :free : :absent,
        # The inverse-Wishart block parameters live on the EMIWBlocks path,
        # which has its own solve method and never reaches optimize_problem.
        phi = :absent,
        r = :absent,
        delta = :absent,
    )
end

"""
    initial_point(stats, fix_rho, estimate_rho_eps, n_omega, estimate_eta, init; rho_limit)

Assemble the optimizer's starting vector in the free-parameter layout: the
`rho` slot (absent under `fix_rho`), `log sigma_a`, `log sigma_z`,
`log sigma_epsilon`, then whichever of `rho_eps`, the `log omega` ladder, or
`eta` this fit estimates. `init` is a NamedTuple in *original outcome units*
(see [`rescale_init_sigmas`](@ref)); `init=nothing` reproduces the heuristic
defaults exactly.

Split out of `optimize_problem` because the starting point is otherwise
unobservable: `theta_unconstrained` on the result is the *minimizer*, and
Nelder-Mead builds its full simplex even at `optim_iters=1`.
"""
function initial_point(
    stats::BipartiteGMRFStats,
    fix_rho::Union{Nothing,Float64},
    estimate_rho_eps::Bool,
    n_omega::Int,
    estimate_eta::Bool,
    init::Union{Nothing,NamedTuple};
    rho_limit::Real=0.99,
)
    status = init_status(stats, fix_rho, estimate_rho_eps, n_omega, estimate_eta)
    # Validate what the caller actually wrote, so the message quotes their
    # numbers; the sigma rescaling that follows preserves positivity.
    validate_init(init, status; rho_limit=rho_limit)
    scaled = rescale_init_sigmas(init, stats.y_std)
    p0 = initial_params(fix_rho, estimate_rho_eps; rho_limit=rho_limit, init=scaled)
    n_omega > 0 && append!(p0, init_omega_codes(scaled, n_omega))
    estimate_eta && push!(p0, init_eta_code(scaled))
    return p0
end

function optimize_problem(
    model::AbstractBipartiteModel,
    stats::BipartiteGMRFStats,
    solver::AbstractGMRFSolver;
    fix_rho::Union{Nothing,Float64}=nothing,
    init::Union{Nothing,NamedTuple}=nothing,
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
    estimate_eta = stats.error_ar1 !== nothing && stats.error_ar1.eta_fixed === nothing
    n_omega = stats.error_classes === nothing ? 0 : length(stats.error_classes.counts) - 1
    p0 = initial_point(stats, fix_rho, estimate_rho_eps, n_omega, estimate_eta, init;
        rho_limit=limit)
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
    nm = NelderMead(initial_simplex=nelder_simplexer(solver))
    elapsed = @elapsed res = optimize(obj, p0, nm, opts)
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
    elseif stats.error_ar1 !== nothing
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
        eta = stats.error_ar1 === nothing ? nothing :
            (stats.error_ar1.eta_fixed === nothing ? eta_from_unconstrained(pfull[5]) :
             stats.error_ar1.eta_fixed),
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
          fix_rho=nothing, init=nothing, seed=42, verbose=false)

Fit a bipartite GMRF by maximum likelihood with the given solver. Extends
`CommonSolve.solve`, so it composes with `using LinearSolve` or other
SciML-style packages without name clashes. [`fit_mle`](@ref) is the
higher-level entry point and delegates here.

`fix_rho` fixes the local-dependence parameter during optimization. `init`
warm-starts the optimizer — see [`fit_mle`](@ref) for its fields and units. Use
[`decompose`](@ref) on the returned result for variance decompositions.
"""
function solve(
    model::AbstractBipartiteModel,
    stats::BipartiteGMRFStats,
    solver::AbstractGMRFSolver;
    fix_rho::Union{Nothing,Float64}=nothing,
    init::Union{Nothing,NamedTuple}=nothing,
    seed::Int=42,
    verbose::Bool=false,
)
    fit = optimize_problem(model, stats, solver;
        fix_rho=fix_rho, init=init, seed=seed, verbose=verbose)
    return build_gmrf_result(fit, solver, fix_rho)
end

"""
    solve(model::BipartiteVarianceStableModel, stats::BipartiteGMRFStats, solver::EMIWBlocks;
          init=nothing, seed=42, verbose=false)

Fit the inverse-Wishart error-block model by variational EM, maximizing the
ELBO rather than the exact likelihood. `fix_rho` is not supported here. `init`
warm-starts `rho`, `sigma_a`, `sigma_z` and the error scale (`phi`, or
`sigma_epsilon` converted through `omega_bar = phi*delta/(delta-2)`); `r` and
`delta` are re-estimated by bracketed searches, so passing them seeds only the
first E-step. See [`fit_mle`](@ref) for the full `init` contract.
"""
function solve(
    model::BipartiteVarianceStableModel,
    stats::BipartiteGMRFStats,
    solver::EMIWBlocks;
    fix_rho::Union{Nothing,Float64}=nothing,
    init::Union{Nothing,NamedTuple}=nothing,
    seed::Int=42,
    verbose::Bool=false,
)
    fix_rho === nothing ||
        throw(ArgumentError("fix_rho is not yet supported for EMIWBlocks."))
    validate_capability(model, stats, solver)
    fit = optimize_emiw(model, stats, solver; verbose=verbose, init=init)
    return build_emiw_result(fit, solver, fix_rho)
end
