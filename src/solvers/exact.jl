# ═══════════════════════════════════════════════════════════════════════════
# ExactCholesky NLL — workspace-based
# ═══════════════════════════════════════════════════════════════════════════

"""
Pre-allocated workspaces for the ExactCholesky optimization loop.
Symbolic factorization is done once; numeric refactorization per iteration.
"""
struct ExactWorkspace
    ws_Q::GaussianMarkovRandomFields.GMRFWorkspace
    ws_M::GaussianMarkovRandomFields.GMRFWorkspace
end

function make_exact_workspace(model::AbstractBipartiteModel, stats::BipartiteGMRFStats)
    # Build Q and M at reference parameters for symbolic factorization.
    # Use a safe rho within the model's limit.
    rho_ref = min(0.1, 0.5 * rho_limit(model))
    # Perturb off exact reciprocals of integers: when a match has
    # F_s·M_s = 1/rho_ref, the Q and VtV off-diagonal entries cancel
    # exactly, Julia's sparse + drops the zero, and update_precision!
    # fails at other rho values due to sparsity pattern mismatch (#107).
    rho_ref += eps(rho_ref)
    Q0 = model_precision(model, rho_ref, 1.0, 1.0)
    M0 = Q0 + stats.design.VtV  # λ=1 at reference
    return ExactWorkspace(
        GaussianMarkovRandomFields.GMRFWorkspace(Q0),
        GaussianMarkovRandomFields.GMRFWorkspace(M0),
    )
end

make_nll_cache(::ExactCholesky, model::AbstractBipartiteModel, stats::BipartiteGMRFStats) =
    make_exact_workspace(model, stats)

function nll_exact_value(
    model::AbstractBipartiteModel,
    stats::BipartiteGMRFStats,
    params_full::Vector{Float64},
    obs::ObservationStats,
    ew::ExactWorkspace,
)
    p = unpack_params(params_full; rho_limit=rho_limit(model))
    all(isfinite, (p.rho, p.sigma_a, p.sigma_z, p.sigma_epsilon)) || return BIG_NLL
    p.sigma_a > 0 && p.sigma_z > 0 && p.sigma_epsilon > 0 || return BIG_NLL
    lambda = 1.0 / p.sigma_epsilon^2

    try
        Q = model_precision(model, p.rho, p.sigma_a, p.sigma_z)
        M = Q + lambda .* obs.design.VtV
        GaussianMarkovRandomFields.update_precision!(ew.ws_Q, Q)
        GaussianMarkovRandomFields.ensure_numeric!(ew.ws_Q)
        GaussianMarkovRandomFields.update_precision!(ew.ws_M, M)
        GaussianMarkovRandomFields.ensure_numeric!(ew.ws_M)
    catch e
        e isa InterruptException && rethrow()
        return BIG_NLL
    end

    ldQ = -GaussianMarkovRandomFields.logdet_cov(ew.ws_Q)  # logdet_cov returns -logdet(Q)
    ldM = -GaussianMarkovRandomFields.logdet_cov(ew.ws_M)
    isfinite(ldQ) && isfinite(ldM) || return BIG_NLL

    x = GaussianMarkovRandomFields.workspace_solve(ew.ws_M, obs.design.projected_y)
    quad = dot(obs.design.projected_y, x)
    isfinite(quad) || return BIG_NLL

    mean_corr = 0.0
    if obs.mean_stats !== nothing
        try
            solve_M = v -> GaussianMarkovRandomFields.workspace_solve(ew.ws_M, v)
            mean_corr, _ = mean_profile_correction(obs.mean_stats, lambda,
                obs.design.projected_y, solve_M)
        catch e
            e isa InterruptException && rethrow()
            return BIG_NLL
        end
    end

    rcorr = residual_corr_term(stats, p.sigma_epsilon, obs.rho_eps)
    rcorr == BIG_NLL && return BIG_NLL
    val = 0.5 * (
        stats.K * 2.0 * log(p.sigma_epsilon) - obs.weights.log_weight_sum +
        (ldM - ldQ) + lambda * obs.design.ydot - lambda^2 * quad - mean_corr + rcorr
    )
    return finite_or_big(val)
end

# Workspace-free convenience wrapper (dense-reference tests, one-off values).
function nll_exact_value(
    model::AbstractBipartiteModel,
    stats::BipartiteGMRFStats,
    params_full::Vector{Float64},
    obs::ObservationStats,
)
    return nll_exact_value(model, stats, params_full, obs, make_exact_workspace(model, stats))
end

nll_value(::ExactCholesky, model, stats, params_full, obs, cache::ExactWorkspace; seed) =
    nll_exact_value(model, stats, params_full, obs, cache)

# The exact objective is smooth but Nelder-Mead's simplex-gradient estimate
# is noisy near the optimum; a loose 1e-3 stopping threshold hands over to
# the L-BFGS polish stage early instead of letting the simplex stall.
nelder_g_abstol(::ExactCholesky, g_rel::Float64) = max(1e-3, g_rel)

function polish(solver::ExactCholesky, obj, res, verbose::Bool)
    (solver.polish && solver.autodiff == :finitediff) || return res, 0.0
    p_start = Vector{Float64}(minimizer(res))
    function fg!(F, G, x)
        if G !== nothing
            finite_difference_gradient!(G, obj, x)
        end
        return F === nothing ? nothing : obj(x)
    end
    polish_opts = Options(iterations=solver.optim_iters, show_trace=verbose,
                          f_reltol=solver.g_reltol)
    elapsed = @elapsed begin
        polished = try
            optimize(only_fg!(fg!), p_start, LBFGS(), polish_opts)
        catch
            nothing
        end
        if polished !== nothing && optim_minimum(polished) <= optim_minimum(res)
            res = polished
        end
    end
    return res, elapsed
end
