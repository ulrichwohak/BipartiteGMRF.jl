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

# Rebuild `A` on `target`'s sparsity pattern, filling 0.0 at structural
# positions that `A` lacks. `A`'s pattern must be a subset of `target`'s, and
# both must be column-sorted CSC. Used by the AR(1) error model to restore the
# zeroed worker-worker off-diagonal slots that sparse `+` drops at eta = 0.
function _align_to_pattern(A::SparseMatrixCSC{Float64,Int}, target::SparseMatrixCSC{Float64,Int})
    vals = zeros(Float64, nnz(target))
    Arv = rowvals(A); Anz = nonzeros(A); Trv = rowvals(target)
    @inbounds for j in 1:size(target, 2)
        Alo = A.colptr[j]; Ahi = A.colptr[j+1] - 1
        Tlo = target.colptr[j]; Thi = target.colptr[j+1] - 1
        a = Alo
        for p in Tlo:Thi
            i = Trv[p]
            while a <= Ahi && Arv[a] < i
                a += 1
            end
            a <= Ahi && Arv[a] == i && (vals[p] = Anz[a])
        end
    end
    return SparseMatrixCSC(target.m, target.n, copy(target.colptr), copy(target.rowval), vals)
end

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
        # AR(1): at eta = 0 the worker-worker off-diagonal slots of V'R^-1 V are
        # zero and get dropped by sparse `+`, shrinking the numeric pattern below
        # the fixed symbolic factorization (built at a nonzero reference eta).
        # Restore them as explicit zeros so update_precision! matches.
        if stats.error_ar1 !== nothing && nnz(M) != nnz(ew.ws_M.Q)
            M = _align_to_pattern(M, ew.ws_M.Q)
        end
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

nelder_simplexer(solver::ExactCholesky) =
    AffineSimplexer(solver.simplex_shift, solver.simplex_scale)

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
