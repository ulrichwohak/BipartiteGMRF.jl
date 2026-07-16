function validate_capability(problem::GMRFProblem, solver::AbstractGMRFSolver)
    if problem.prior isa VarianceStablePrior
        problem.weighting.observations == :raw ||
            throw(ArgumentError("VarianceStablePrior currently supports only raw observation weighting."))
    end
    if problem.prior isa SpectralPrior && solver isa ExactCholesky
        throw(ArgumentError("ExactCholesky for SpectralPrior is not implemented; use HutchSLQ."))
    end
    return nothing
end

# ─── Model construction from problem ──────────────────────────────────────

"""
Build an `AbstractBipartiteModel` from a `GMRFProblem`'s prior spec and adjacency.
Cached on first call per problem via the metadata field.
"""
function _build_model(problem::GMRFProblem)
    return to_model(problem.prior, problem.A_prior)
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

# ─── Legacy wrappers (used by operators/decomposition until migrated) ─────

function precision_matrix(problem::GMRFProblem, rho::Float64, sigma_a::Float64, sigma_z::Float64)
    model = _build_model(problem)
    return model_precision(model, rho, sigma_a, sigma_z)
end

function posterior_precision_matrix(
    problem::GMRFProblem,
    rho::Float64,
    sigma_a::Float64,
    sigma_z::Float64,
    sigma_epsilon::Float64,
)
    Q = precision_matrix(problem, rho, sigma_a, sigma_z)
    return Q + (1.0 / sigma_epsilon^2) .* problem.VtV
end

# ─── Matrix-free operators (HutchSLQ path, unchanged) ─────────────────────

function q_operator(problem::GMRFProblem, rho::Float64, sigma_a::Float64, sigma_z::Float64)
    if problem.prior isa VarianceStablePrior
        return make_qop_vs(problem, rho, sigma_a, sigma_z)
    end
    return make_qop(problem, rho, sigma_a, sigma_z)
end

function q_diag(problem::GMRFProblem, rho::Float64, sigma_a::Float64, sigma_z::Float64)
    inv_sa2 = 1.0 / sigma_a^2
    inv_sz2 = 1.0 / sigma_z^2
    n = problem.N_firms + problem.N_workers
    out = Vector{Float64}(undef, n)
    if problem.prior isa VarianceStablePrior
        @inbounds for i in 1:problem.N_firms
            out[i] = (1.0 + rho^2 * (problem.d_f[i] - 1.0)) * inv_sa2
        end
        @inbounds for j in 1:problem.N_workers
            out[problem.N_firms + j] = (1.0 + rho^2 * (problem.d_w[j] - 1.0)) * inv_sz2
        end
    else
        @inbounds for i in 1:problem.N_firms
            out[i] = problem.diag_f[i] * inv_sa2
        end
        @inbounds for j in 1:problem.N_workers
            out[problem.N_firms + j] = problem.diag_w[j] * inv_sz2
        end
    end
    return out
end

# ─── Observation stats ─────────────────────────────────────────────────────

function objective_stats(problem::GMRFProblem, params_full::Vector{Float64})
    if problem.weighting.observations == :effective && problem.weighting.rho_eps == :estimate
        rho_eps = rhoeps_from_unconstrained(params_full[5])
        stats = build_match_weight_stats(
            problem.base_f_rows,
            problem.base_w_cols,
            problem.base_y,
            problem.base_T,
            problem.N_firms,
            problem.N_workers,
            rho_eps,
        )
        return merge(stats, (rho_eps = rho_eps,))
    end
    return (
        ydot = problem.ydot,
        projected_y = problem.projected_y,
        VtV = problem.VtV,
        cnt_f = problem.cnt_f,
        cnt_w = problem.cnt_w,
        A_obs = problem.A_obs,
        At_obs = problem.At_obs,
        log_weight_sum = problem.log_weight_sum,
        effective_weight_sum = problem.effective_weight_sum,
        mean_effective_weight = problem.mean_effective_weight,
        max_effective_weight = problem.max_effective_weight,
        effective_weight_over_T_sum = problem.effective_weight_over_T_sum,
        rho_eps = problem.rho_eps_likelihood,
    )
end

function residual_corr_term(problem::GMRFProblem, sigma_epsilon::Float64, rho_eps::Union{Nothing,Float64})
    rho_eps === nothing && return 0.0
    0.0 <= rho_eps < 1.0 || return BIG_NLL
    omr = 1.0 - rho_eps
    lambda = 1.0 / sigma_epsilon^2
    return Float64(problem.within_df) * (2.0 * log(sigma_epsilon) + log(omr)) +
        lambda * Float64(problem.within_ss) / omr
end

# ═══════════════════════════════════════════════════════════════════════════
# ExactCholesky NLL — workspace-based
# ═══════════════════════════════════════════════════════════════════════════

"""
Pre-allocated workspaces for the ExactCholesky optimization loop.
Symbolic factorization is done once; numeric refactorization per iteration.
"""
mutable struct ExactWorkspace
    model::AbstractBipartiteModel
    ws_Q::GaussianMarkovRandomFields.GMRFWorkspace
    ws_M::GaussianMarkovRandomFields.GMRFWorkspace
end

function make_exact_workspace(problem::GMRFProblem)
    model = _build_model(problem)
    # Build Q and M at reference parameters for symbolic factorization.
    # Use a safe rho within the model's limit.
    rho_ref = min(0.1, 0.5 * rho_limit(problem.prior))
    Q0 = model_precision(model, rho_ref, 1.0, 1.0)
    M0 = Q0 + problem.VtV  # λ=1 at reference
    ws_Q = GaussianMarkovRandomFields.GMRFWorkspace(Q0)
    ws_M = GaussianMarkovRandomFields.GMRFWorkspace(M0)
    return ExactWorkspace(model, ws_Q, ws_M)
end

function nll_exact_value(problem::GMRFProblem, params_full::Vector{Float64}, stats, ew::ExactWorkspace)
    p = unpack_params(params_full; rho_limit=rho_limit(problem.prior))
    all(isfinite, (p.rho, p.sigma_a, p.sigma_z, p.sigma_epsilon)) || return BIG_NLL
    p.sigma_a > 0 && p.sigma_z > 0 && p.sigma_epsilon > 0 || return BIG_NLL
    lambda = 1.0 / p.sigma_epsilon^2

    Q = model_precision(ew.model, p.rho, p.sigma_a, p.sigma_z)
    M = Q + lambda .* stats.VtV

    try
        GaussianMarkovRandomFields.update_precision!(ew.ws_Q, Q)
        GaussianMarkovRandomFields.ensure_numeric!(ew.ws_Q)
    catch
        return BIG_NLL
    end

    try
        GaussianMarkovRandomFields.update_precision!(ew.ws_M, M)
        GaussianMarkovRandomFields.ensure_numeric!(ew.ws_M)
    catch
        return BIG_NLL
    end

    ldQ = -GaussianMarkovRandomFields.logdet_cov(ew.ws_Q)  # logdet_cov returns -logdet(Q)
    ldM = -GaussianMarkovRandomFields.logdet_cov(ew.ws_M)
    isfinite(ldQ) && isfinite(ldM) || return BIG_NLL

    # Solve M \ projected_y using workspace's Cholesky factor
    x = ew.ws_M.backend.factor \ stats.projected_y
    quad = dot(stats.projected_y, x)
    isfinite(quad) || return BIG_NLL

    rcorr = residual_corr_term(problem, p.sigma_epsilon, stats.rho_eps)
    rcorr == BIG_NLL && return BIG_NLL
    val = 0.5 * (
        problem.K * 2.0 * log(p.sigma_epsilon) - Float64(stats.log_weight_sum) +
        (ldM - ldQ) + lambda * stats.ydot - lambda^2 * quad + rcorr
    )
    return finite_or_big(val)
end

# Legacy fallback (no workspace)
function nll_exact_value(problem::GMRFProblem, params_full::Vector{Float64}, stats)
    p = unpack_params(params_full; rho_limit=rho_limit(problem.prior))
    all(isfinite, (p.rho, p.sigma_a, p.sigma_z, p.sigma_epsilon)) || return BIG_NLL
    p.sigma_a > 0 && p.sigma_z > 0 && p.sigma_epsilon > 0 || return BIG_NLL
    lambda = 1.0 / p.sigma_epsilon^2
    Q = precision_matrix(problem, p.rho, p.sigma_a, p.sigma_z)
    M = Q + lambda .* stats.VtV
    FQ = try
        cholesky(Symmetric(Q))
    catch
        return BIG_NLL
    end
    FM = try
        cholesky(Symmetric(M))
    catch
        return BIG_NLL
    end
    ldQ = logdet(FQ)
    ldM = logdet(FM)
    isfinite(ldQ) && isfinite(ldM) || return BIG_NLL
    x = FM \ stats.projected_y
    quad = dot(stats.projected_y, x)
    isfinite(quad) || return BIG_NLL
    rcorr = residual_corr_term(problem, p.sigma_epsilon, stats.rho_eps)
    rcorr == BIG_NLL && return BIG_NLL
    val = 0.5 * (
        problem.K * 2.0 * log(p.sigma_epsilon) - Float64(stats.log_weight_sum) +
        (ldM - ldQ) + lambda * stats.ydot - lambda^2 * quad + rcorr
    )
    return finite_or_big(val)
end

# ═══════════════════════════════════════════════════════════════════════════
# HutchSLQ NLL — unchanged (matrix-free, no workspace)
# ═══════════════════════════════════════════════════════════════════════════

mutable struct HutchCache{Q<:Union{QOp,QOpVS}}
    dV::Vector{Float64}
    Mdiag::Vector{Float64}
    pcg::PCGWorkspace
    slqQ::SLQWorkspace
    slqM::SLQWorkspace
    qop::Q
    mop::MOp{Q}
end

mutable struct VSHutchCache
    dV::Vector{Float64}
    Mdiag::Vector{Float64}
    pcg::PCGWorkspace
    slqB::SLQWorkspace
    slqK::SLQWorkspace
    qop::QOpVS
    mop::MOp{QOpVS}
    bop::QOpVS
    kop::ScaledMOp{QOpVS}
end

function make_hutch_cache(problem::GMRFProblem, solver::HutchSLQ)
    n = problem.N_firms + problem.N_workers
    qop = q_operator(problem, 0.0, 1.0, 1.0)
    mop = MOp(qop, problem.VtV, zeros(n), 1.0)
    if problem.prior isa VarianceStablePrior
        bop = make_qop_vs(problem, 0.0, 1.0, 1.0)
        kop = ScaledMOp(bop, problem.VtV, ones(n), zeros(n), zeros(n), 1.0)
        return VSHutchCache(
            Vector{Float64}(diag(problem.VtV)),
            zeros(n),
            PCGWorkspace(n),
            SLQWorkspace(n, solver.lanczos_iters),
            SLQWorkspace(n, solver.lanczos_iters),
            qop,
            mop,
            bop,
            kop,
        )
    end
    return HutchCache(
        Vector{Float64}(diag(problem.VtV)),
        zeros(n),
        PCGWorkspace(n),
        SLQWorkspace(n, solver.lanczos_iters),
        SLQWorkspace(n, solver.lanczos_iters),
        qop,
        mop,
    )
end

function hutch_logdet_difference!(
    cache::HutchCache,
    solver::HutchSLQ,
    n::Int,
    seed::Int,
    ::NamedTuple,
    ::Float64,
)
    ldQ = slq_logdet_spd_mul_cached!(cache.qop, n, cache.slqQ;
        m=solver.logdet_probes, k=solver.lanczos_iters, seed=seed)
    ldM = slq_logdet_spd_mul_cached!(cache.mop, n, cache.slqM;
        m=solver.logdet_probes, k=solver.lanczos_iters, seed=seed + 10_000)
    return ldM - ldQ
end

function hutch_logdet_difference!(
    cache::VSHutchCache,
    solver::HutchSLQ,
    n::Int,
    seed::Int,
    p::NamedTuple,
    lambda::Float64,
)
    set_q_params!(cache.bop, p.rho, 1.0, 1.0)
    cache.kop.lambda = lambda
    cache.kop.VtV = cache.mop.VtV
    nf = cache.bop.n_firms
    @views fill!(cache.kop.scale[1:nf], p.sigma_a)
    @views fill!(cache.kop.scale[(nf + 1):n], p.sigma_z)

    ldB = slq_logdet_spd_mul_cached!(cache.bop, n, cache.slqB;
        m=solver.logdet_probes, k=solver.lanczos_iters, seed=seed)
    ldK = slq_logdet_spd_mul_cached!(cache.kop, n, cache.slqK;
        m=solver.logdet_probes, k=solver.lanczos_iters, seed=seed)
    return ldK - ldB
end

function set_q_params!(qop::QOp, rho::Float64, sigma_a::Float64, sigma_z::Float64)
    qop.inv_sa2 = 1.0 / sigma_a^2
    qop.inv_sz2 = 1.0 / sigma_z^2
    qop.cross = rho / (sigma_a * sigma_z)
    return qop
end

function set_q_params!(qop::QOpVS, rho::Float64, sigma_a::Float64, sigma_z::Float64)
    qop.inv_sa2 = 1.0 / sigma_a^2
    qop.inv_sz2 = 1.0 / sigma_z^2
    qop.cross = rho / (sigma_a * sigma_z)
    qop.rho_sq = rho^2
    return qop
end

function nll_hutch_value(
    problem::GMRFProblem,
    solver::HutchSLQ,
    params_full::Vector{Float64},
    stats,
    cache::Union{HutchCache,VSHutchCache};
    seed::Int,
)
    p = unpack_params(params_full; rho_limit=rho_limit(problem.prior))
    all(isfinite, (p.rho, p.sigma_a, p.sigma_z, p.sigma_epsilon)) || return BIG_NLL
    p.sigma_a > 0 && p.sigma_z > 0 && p.sigma_epsilon > 0 || return BIG_NLL

    lambda = 1.0 / p.sigma_epsilon^2
    set_q_params!(cache.qop, p.rho, p.sigma_a, p.sigma_z)
    cache.mop.lambda = lambda
    cache.mop.VtV = stats.VtV
    cache.dV .= Vector{Float64}(diag(stats.VtV))
    Qdiag = q_diag(problem, p.rho, p.sigma_a, p.sigma_z)
    @. cache.Mdiag = Qdiag + lambda * cache.dV

    x, ok, _, _ = pcg_solve!(cache.pcg, cache.mop, stats.projected_y;
        tol=solver.cg_tol, maxiter=solver.cg_maxiter, Mdiag=cache.Mdiag)
    ok || return BIG_NLL
    quad = dot(stats.projected_y, x)
    isfinite(quad) || return BIG_NLL

    n = length(stats.projected_y)
    ld_difference = hutch_logdet_difference!(cache, solver, n, seed, p, lambda)
    isfinite(ld_difference) || return BIG_NLL
    rcorr = residual_corr_term(problem, p.sigma_epsilon, stats.rho_eps)
    rcorr == BIG_NLL && return BIG_NLL
    val = 0.5 * (
        problem.K * 2.0 * log(p.sigma_epsilon) - Float64(stats.log_weight_sum) +
        ld_difference + lambda * stats.ydot - lambda^2 * quad + rcorr
    )
    return finite_or_big(val)
end

# ═══════════════════════════════════════════════════════════════════════════
# Optimization loop
# ═══════════════════════════════════════════════════════════════════════════

function optimize_problem(
    problem::GMRFProblem,
    solver::AbstractGMRFSolver;
    fix_rho::Union{Nothing,Float64}=nothing,
    seed::Int=42,
    verbose::Bool=false,
)
    validate_capability(problem, solver)
    limit = rho_limit(problem.prior)
    if fix_rho !== nothing && !(abs(fix_rho) < limit)
        throw(ArgumentError("fix_rho must lie in (-$(limit), $(limit))."))
    end

    estimate_rho_eps = problem.weighting.observations == :effective &&
        problem.weighting.rho_eps == :estimate
    p0 = initial_params(fix_rho, estimate_rho_eps; rho_limit=limit)
    evals = Ref(0)

    # Pre-allocate solver-specific caches
    exact_ws = solver isa ExactCholesky ? make_exact_workspace(problem) : nothing
    hutch_cache = solver isa HutchSLQ ? make_hutch_cache(problem, solver) : nothing

    function obj(pfree)
        evals[] += 1
        pfull = full_params(Vector{Float64}(pfree), fix_rho, estimate_rho_eps; rho_limit=limit)
        stats = objective_stats(problem, pfull)
        if solver isa ExactCholesky
            return nll_exact_value(problem, pfull, stats, exact_ws)
        else
            return nll_hutch_value(problem, solver, pfull, stats, hutch_cache; seed=seed)
        end
    end

    iterations = solver.optim_iters
    f0 = obj(p0)
    fscale = (isfinite(f0) && f0 < BIG_NLL) ? max(1.0, abs(f0)) : 1.0
    g_rel = solver.g_reltol * fscale
    g_abstol = solver isa ExactCholesky ? max(1e-3, g_rel) : g_rel
    opts = Options(iterations=iterations, show_trace=verbose, g_tol=g_abstol)
    elapsed = @elapsed res = optimize(obj, p0, NelderMead(), opts)
    if solver isa ExactCholesky && solver.polish && solver.autodiff == :finitediff
        p_start = Vector{Float64}(minimizer(res))
        function fg!(F, G, x)
            if G !== nothing
                finite_difference_gradient!(G, obj, x)
            end
            return F === nothing ? nothing : obj(x)
        end
        polish_opts = Options(iterations=solver.optim_iters, show_trace=verbose,
                              f_reltol=solver.g_reltol)
        polish_elapsed = @elapsed begin
            polished = try
                optimize(only_fg!(fg!), p_start, LBFGS(), polish_opts)
            catch
                nothing
            end
            if polished !== nothing && optim_minimum(polished) <= optim_minimum(res)
                res = polished
            end
        end
        elapsed += polish_elapsed
    end
    pfree = Vector{Float64}(minimizer(res))
    pfull = full_params(pfree, fix_rho, estimate_rho_eps; rho_limit=limit)
    stats = objective_stats(problem, pfull)
    final_problem = estimate_rho_eps ? with_observation_stats(problem, stats, stats.rho_eps) : problem
    val = obj(pfree)
    decoded = unpack_params(pfull; rho_limit=limit)
    rho_eps = estimate_rho_eps ? stats.rho_eps : problem.rho_eps_likelihood
    return (
        rho = decoded.rho,
        sigma_a = decoded.sigma_a * final_problem.y_std,
        sigma_z = decoded.sigma_z * final_problem.y_std,
        sigma_epsilon = decoded.sigma_epsilon * final_problem.y_std,
        rho_eps = rho_eps,
        nll = val,
        converged = optim_converged(res),
        iterations = optim_iterations(res),
        obj_evals = evals[],
        optimization_time = elapsed,
        theta_unconstrained = pfull,
        problem = final_problem,
    )
end
