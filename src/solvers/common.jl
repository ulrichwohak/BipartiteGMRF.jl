function validate_capability(problem::GMRFProblem, solver::AbstractGMRFSolver)
    if problem.prior isa VarianceStablePrior
        problem.weighting.observations == :raw ||
            throw(ArgumentError("VarianceStablePrior currently supports only raw observation weighting."))
        # ExactCholesky is exact and cheap on acyclic graphs (no fill-in) and
        # avoids the HutchSLQ log-det cancellation that corrupts the VS objective at
        # small sigma_z (see issue #82). It is also permitted on *cyclic* graphs:
        # the VS precision is positive-definite while |rho| * lambda_NB < 1 (the
        # caller enforces this via rho_limit), an indefinite precision is rejected
        # per-evaluation (the cholesky in nll_exact_value returns BIG_NLL on
        # failure), and fill-in is acceptable on mildly cyclic / NB-pruned graphs.
        # Cyclic VS input is already warned about at construction (prepare.jl), and
        # strict_forest=true errors there. An automatic safeguard that sets/checks
        # rho_limit from lambda_NB is tracked in issues #79 (spectral analysis) and
        # #78 (graph pruning).
    end
    if problem.prior isa SpectralPrior && solver isa ExactCholesky
        throw(ArgumentError("ExactCholesky for SpectralPrior is not implemented; use HutchSLQ."))
    end
    return nothing
end

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

function precision_matrix(problem::GMRFProblem, rho::Float64, sigma_a::Float64, sigma_z::Float64)
    inv_sa2 = 1.0 / sigma_a^2
    inv_sz2 = 1.0 / sigma_z^2
    cross = rho / (sigma_a * sigma_z)
    nf = problem.N_firms
    nw = problem.N_workers

    if problem.prior isa VarianceStablePrior
        diag_f = (1.0 .+ rho^2 .* (problem.d_f .- 1.0)) .* inv_sa2
        diag_w = (1.0 .+ rho^2 .* (problem.d_w .- 1.0)) .* inv_sz2
        W = problem.A_prior
        Wt = problem.At_prior
    else
        diag_f = problem.diag_f .* inv_sa2
        diag_w = problem.diag_w .* inv_sz2
        W = spdiagm(0 => problem.df_is) * problem.A_prior * spdiagm(0 => problem.dw_is)
        Wt = copy(transpose(W))
    end

    return [spdiagm(0 => diag_f) (-cross .* W); (-cross .* Wt) spdiagm(0 => diag_w)]
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

mutable struct HutchCache{Q<:Union{QOp,QOpVS}}
    dV::Vector{Float64}
    Mdiag::Vector{Float64}
    pcg::PCGWorkspace
    slqQ::SLQWorkspace
    slqM::SLQWorkspace
    qop::Q
    mop::MOp{Q}
end

function make_hutch_cache(problem::GMRFProblem, solver::HutchSLQ)
    n = problem.N_firms + problem.N_workers
    qop = q_operator(problem, 0.0, 1.0, 1.0)
    mop = MOp(qop, problem.VtV, zeros(n), 1.0)
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
    cache::HutchCache;
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
    ldQ = slq_logdet_spd_mul_cached!(cache.qop, n, cache.slqQ;
        m=solver.logdet_probes, k=solver.lanczos_iters, seed=seed)
    ldM = slq_logdet_spd_mul_cached!(cache.mop, n, cache.slqM;
        m=solver.logdet_probes, k=solver.lanczos_iters, seed=seed + 10_000)
    isfinite(ldQ) && isfinite(ldM) || return BIG_NLL
    rcorr = residual_corr_term(problem, p.sigma_epsilon, stats.rho_eps)
    rcorr == BIG_NLL && return BIG_NLL
    val = 0.5 * (
        problem.K * 2.0 * log(p.sigma_epsilon) - Float64(stats.log_weight_sum) +
        (ldM - ldQ) + lambda * stats.ydot - lambda^2 * quad + rcorr
    )
    return finite_or_big(val)
end

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
    cache = solver isa HutchSLQ ? make_hutch_cache(problem, solver) : nothing

    function obj(pfree)
        evals[] += 1
        pfull = full_params(Vector{Float64}(pfree), fix_rho, estimate_rho_eps; rho_limit=limit)
        stats = objective_stats(problem, pfull)
        if solver isa ExactCholesky
            return nll_exact_value(problem, pfull, stats)
        else
            return nll_hutch_value(problem, solver, pfull, stats, cache; seed=seed)
        end
    end

    iterations = solver.optim_iters
    # NelderMead's convergence metric is the absolute spread of objective values,
    # which grows with the problem size (nll scales with the number of
    # observations), so a fixed absolute tolerance is unreachable on large graphs
    # (see #80, #84). Scale it by the objective magnitude at the start point.
    # HutchSLQ uses the pure relative tolerance; ExactCholesky keeps a 1e-3 floor
    # (its calibrated small-problem behaviour, against which the solver-agreement
    # tests are tuned) and only loosens above it on large problems.
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
        # Stop the polish when the objective stops improving (relative change <
        # g_reltol), not only on the default g_abstol=1e-8: the finite-difference
        # gradient floors well above 1e-8 on large problems, so g_abstol is
        # unreachable and LBFGS would otherwise grind to the iteration cap (#86).
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
