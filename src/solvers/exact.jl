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
    q_values::Vector{Float64}
    m_values::Vector{Float64}
end

ExactWorkspace(ws_Q::GaussianMarkovRandomFields.GMRFWorkspace,
               ws_M::GaussianMarkovRandomFields.GMRFWorkspace) =
    ExactWorkspace(ws_Q, ws_M, zeros(nnz(ws_Q.Q)), zeros(nnz(ws_M.Q)))

# Positive markers describe stored positions, including explicit zeros. They
# are used only to build structural unions, never as numerical precisions.
_exact_pattern(A::SparseMatrixCSC{Float64,Int}) =
    SparseMatrixCSC(A.m, A.n, copy(A.colptr), copy(A.rowval), ones(nnz(A)))

function _exact_prior_pattern(model::AbstractBipartiteModel)
    g = model.graph
    return [
        spdiagm(0 => ones(g.n_firms)) _exact_pattern(g.A)
        _exact_pattern(g.At) spdiagm(0 => ones(g.n_workers))
    ]
end

function make_exact_workspace(model::AbstractBipartiteModel, stats::BipartiteGMRFStats)
    # All supported exact priors have diagonal + bipartite-graph support.
    # Observation support must also be structural: AR1 entries can cancel at
    # ANY reference eta, and Q + VtV can cancel even for independent errors.
    Q_pattern = _exact_prior_pattern(model)
    observation_pattern = stats.error_ar1 === nothing ? stats.design.VtV : stats.error_ar1.pattern
    M0 = Q_pattern + _exact_pattern(observation_pattern)

    # The independent prior (rho=0) is a valid numerical reference even when
    # cyclic VS graphs have a very small feasible rho range. Its zero edge
    # values are retained on the complete pattern for symbolic analysis.
    Q0 = _align_to_pattern(model_precision(model, 0.0, 1.0, 1.0), Q_pattern)
    fill!(nonzeros(M0), 0.0)
    _add_to_pattern!(nonzeros(M0), Q0, M0)
    _add_to_pattern!(nonzeros(M0), stats.design.VtV, M0) # λ=1 at reference
    return ExactWorkspace(
        GaussianMarkovRandomFields.GMRFWorkspace(Q0),
        GaussianMarkovRandomFields.GMRFWorkspace(M0),
    )
end

make_nll_cache(::ExactCholesky, model::AbstractBipartiteModel, stats::BipartiteGMRFStats) =
    make_exact_workspace(model, stats)

# Add scale*A into a value buffer on target's fixed, column-sorted CSC pattern.
# Reject EVERY unsupported stored position (even an explicit zero); silently
# discarding entries would change the model. Missing source positions are fine.
# The caller owns and resets the buffer; on failure it may be partially populated.
function _add_to_pattern!(vals::Vector{Float64}, A::SparseMatrixCSC{Float64,Int},
                          target::SparseMatrixCSC{Float64,Int}, scale::Float64=1.0)
    size(A) == size(target) || throw(DimensionMismatch("source and target precision dimensions differ"))
    length(vals) == nnz(target) || throw(DimensionMismatch("value buffer does not match target pattern"))
    Arv = rowvals(A); Anz = nonzeros(A); Trv = rowvals(target)
    @inbounds for j in 1:size(target, 2)
        p = target.colptr[j]
        last = target.colptr[j+1] - 1
        for a in nzrange(A, j)
            i = Arv[a]
            while p <= last && Trv[p] < i
                p += 1
            end
            p <= last && Trv[p] == i || throw(ArgumentError(
                "precision entry ($i, $j) is outside the fixed sparsity pattern"))
            vals[p] += scale * Anz[a]
        end
    end
    return vals
end

# Allocating convenience wrapper for initialization and pattern validation.
function _align_to_pattern(A::SparseMatrixCSC{Float64,Int}, target::SparseMatrixCSC{Float64,Int})
    vals = zeros(Float64, nnz(target))
    _add_to_pattern!(vals, A, target)
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
    abs(p.rho) < rho_limit(model) || return BIG_NLL
    lambda = 1.0 / p.sigma_epsilon^2
    isfinite(lambda) || return BIG_NLL

    # Validate support BEFORE numerical factorization. Structural mistakes are
    # programming errors, not infeasible parameter trials. Assemble directly
    # into reusable buffers to avoid allocating a sparse posterior each time.
    Q = model_precision(model, p.rho, p.sigma_a, p.sigma_z)
    fill!(ew.q_values, 0.0)
    fill!(ew.m_values, 0.0)
    _add_to_pattern!(ew.q_values, Q, ew.ws_Q.Q)
    _add_to_pattern!(ew.m_values, Q, ew.ws_M.Q)
    _add_to_pattern!(ew.m_values, obs.design.VtV, ew.ws_M.Q, lambda)
    all(isfinite, ew.q_values) && all(isfinite, ew.m_values) || return BIG_NLL
    GaussianMarkovRandomFields.update_precision_values!(ew.ws_Q, ew.q_values)
    GaussianMarkovRandomFields.update_precision_values!(ew.ws_M, ew.m_values)

    try
        GaussianMarkovRandomFields.ensure_numeric!(ew.ws_Q)
        GaussianMarkovRandomFields.ensure_numeric!(ew.ws_M)
        # The workspace's CHOLMOD backend refactorizes with check=false.
        # A failed factorization therefore need not throw an exception.
        issuccess(ew.ws_Q.backend.factor) && issuccess(ew.ws_M.backend.factor) || return BIG_NLL
    catch e
        e isa PosDefException && return BIG_NLL
        rethrow()
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
            e isa PosDefException && return BIG_NLL
            rethrow()
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
        catch e
            # Preserve the numerical/line-search fallback, but never conceal
            # the structural checks above (or an explicit interruption).
            (e isa ArgumentError || e isa DimensionMismatch || e isa InterruptException) && rethrow()
            nothing
        end
        if polished !== nothing && optim_minimum(polished) <= optim_minimum(res)
            res = polished
        end
    end
    return res, elapsed
end
