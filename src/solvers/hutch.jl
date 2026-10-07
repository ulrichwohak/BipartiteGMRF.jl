# ═══════════════════════════════════════════════════════════════════════════
# HutchSLQ NLL — matrix-free
# ═══════════════════════════════════════════════════════════════════════════

mutable struct HutchCache{Q<:Union{QOp,QOpVS}}
    dV::Vector{Float64}
    Mdiag::Vector{Float64}
    pcg::PCGWorkspace
    slqQ::SLQWorkspace
    slqM::SLQWorkspace
    qop::Q
    mop::MOp{Q}
    mean::Union{Nothing,MeanProfileWorkspace}
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
    mean::Union{Nothing,MeanProfileWorkspace}
end

function make_hutch_cache(model::AbstractBipartiteModel, stats::BipartiteGMRFStats, solver::HutchSLQ)
    n = model.graph.n_firms + model.graph.n_workers
    qop = q_operator(model, 0.0, 1.0, 1.0)
    mop = MOp(qop, stats.design.VtV, zeros(n), 1.0)
    if model isa BipartiteVarianceStableModel
        bop = make_qop_vs(model, 0.0, 1.0, 1.0)
        kop = ScaledMOp(bop, stats.design.VtV, ones(n), zeros(n), zeros(n), 1.0)
        return VSHutchCache(
            Vector{Float64}(diag(stats.design.VtV)),
            zeros(n),
            PCGWorkspace(n),
            SLQWorkspace(n, solver.lanczos_iters),
            SLQWorkspace(n, solver.lanczos_iters),
            qop,
            mop,
            bop,
            kop,
            stats.mean_stats === nothing ? nothing : MeanProfileWorkspace(stats.mean_stats),
        )
    end
    return HutchCache(
        Vector{Float64}(diag(stats.design.VtV)),
        zeros(n),
        PCGWorkspace(n),
        SLQWorkspace(n, solver.lanczos_iters),
        SLQWorkspace(n, solver.lanczos_iters),
        qop,
        mop,
        stats.mean_stats === nothing ? nothing : MeanProfileWorkspace(stats.mean_stats),
    )
end

make_nll_cache(solver::HutchSLQ, model::AbstractBipartiteModel, stats::BipartiteGMRFStats) =
    make_hutch_cache(model, stats, solver)

function objective_parameters_valid(::HutchSLQ, stats::BipartiteGMRFStats,
    params_full::Vector{Float64})
    all(isfinite, params_full) || return false
    ar = stats.error_ar1
    ar === nothing && return true
    eta = ar.eta_fixed === nothing ? eta_from_unconstrained(params_full[5]) : ar.eta_fixed
    # The three-component AR assembly subtracts terms amplified by
    # 1/(1-eta^2). At this threshold cancellation can consume at least half
    # Float64's significant digits, even for an eta-independent singleton
    # product. Reject numerically unresolved trials; do not clamp eta to a
    # different fitted model or rely on a small PCG residual to detect damaged
    # statistics. ExactCholesky retains its existing finite-eta behavior.
    return 1.0 - eta^2 > sqrt(eps(Float64))
end

# Q and M log-determinants use independent probe streams (seed offset); the
# VS path instead evaluates B and K with common random numbers, which reduces
# the variance of the ldK - ldB difference where the congruence scaling makes
# the two spectra comparable.
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
    rho_sq = rho^2
    inv_one_minus_rho_sq = 1.0 / (1.0 - rho_sq)
    qop.inv_sa2 = inv_one_minus_rho_sq / sigma_a^2
    qop.inv_sz2 = inv_one_minus_rho_sq / sigma_z^2
    qop.cross = rho * inv_one_minus_rho_sq / (sigma_a * sigma_z)
    qop.rho_sq = rho_sq
    return qop
end

# Assemble the diagonal from the same operator used by PCG. This avoids a
# temporary network-sized q_diag vector on every objective evaluation.
function hutch_preconditioner!(out::Vector{Float64}, qop::QOp,
    dV::Vector{Float64}, lambda::Float64)
    nf = qop.n_firms
    @inbounds for i in 1:nf
        out[i] = qop.inv_sa2 * qop.diag_f[i] + lambda * dV[i]
    end
    @inbounds for j in eachindex(qop.diag_w)
        out[nf + j] = qop.inv_sz2 * qop.diag_w[j] + lambda * dV[nf + j]
    end
    return out
end

function hutch_preconditioner!(out::Vector{Float64}, qop::QOpVS,
    dV::Vector{Float64}, lambda::Float64)
    nf = qop.n_firms
    @inbounds for i in 1:nf
        out[i] = (1.0 + qop.rho_sq * (qop.d_f[i] - 1.0)) * qop.inv_sa2 + lambda * dV[i]
    end
    @inbounds for j in eachindex(qop.d_w)
        out[nf + j] = (1.0 + qop.rho_sq * (qop.d_w[j] - 1.0)) * qop.inv_sz2 + lambda * dV[nf + j]
    end
    return out
end

function refresh_hutch_cache!(cache::Union{HutchCache,VSHutchCache},
    obs::ObservationStats, p::NamedTuple)
    lambda = inv(p.sigma_epsilon^2)
    set_q_params!(cache.qop, p.rho, p.sigma_a, p.sigma_z)
    cache.mop.lambda = lambda
    cache.mop.VtV = obs.design.VtV
    # eta and other residual parameters change values, not necessarily sparse
    # support or matrix identity. Read the current diagonal unconditionally;
    # indexed CSC lookup also avoids allocating a network-sized diag vector.
    @inbounds for i in eachindex(cache.dV)
        cache.dV[i] = obs.design.VtV[i, i]
    end
    hutch_preconditioner!(cache.Mdiag, cache.qop, cache.dV, lambda)
    return lambda
end

function hutch_mean_solve!(cache::Union{HutchCache,VSHutchCache},
    solver::HutchSLQ, v::AbstractVector{<:Real})
    solution, ok, iterations, relres = pcg_solve!(cache.pcg, cache.mop, v;
        tol=solver.cg_tol, maxiter=solver.cg_maxiter, Mdiag=cache.Mdiag)
    ok || throw(MeanProfileError(
        "Mean-profile PCG did not converge after $(iterations) iterations " *
        "(relative residual=$(relres), tolerance=$(solver.cg_tol)); " *
        "increase cg_maxiter or improve scaling. No direct-factorization fallback was used."))
    return solution
end

function nll_hutch_value(
    model::AbstractBipartiteModel,
    stats::BipartiteGMRFStats,
    solver::HutchSLQ,
    params_full::Vector{Float64},
    obs::ObservationStats,
    cache::Union{HutchCache,VSHutchCache};
    seed::Int,
)
    objective_parameters_valid(solver, stats, params_full) || return BIG_NLL
    p = unpack_params(params_full; rho_limit=rho_limit(model))
    all(isfinite, (p.rho, p.sigma_a, p.sigma_z, p.sigma_epsilon)) || return BIG_NLL
    p.sigma_a > 0 && p.sigma_z > 0 && p.sigma_epsilon > 0 || return BIG_NLL

    lambda = refresh_hutch_cache!(cache, obs, p)
    isfinite(lambda) && lambda > 0.0 || return BIG_NLL

    x, ok, _, _ = pcg_solve!(cache.pcg, cache.mop, obs.design.projected_y;
        tol=solver.cg_tol, maxiter=solver.cg_maxiter, Mdiag=cache.Mdiag)
    ok || return BIG_NLL
    quad = dot(obs.design.projected_y, x)
    isfinite(quad) || return BIG_NLL

    mean_corr = 0.0
    if obs.mean_stats !== nothing
        ms = obs.mean_stats
        pcg_solve_M = v -> hutch_mean_solve!(cache, solver, v)
        try
            mean_corr, _ = mean_profile_correction(ms, lambda,
                obs.design.projected_y, pcg_solve_M, cache.mean::MeanProfileWorkspace;
                solved_y=x, symmetry_rtol=max(1e-10, 10 * solver.cg_tol))
        catch
            return BIG_NLL
        end
    end

    n = length(obs.design.projected_y)
    ld_difference = hutch_logdet_difference!(cache, solver, n, seed, p, lambda)
    isfinite(ld_difference) || return BIG_NLL
    rcorr = residual_corr_term(stats, p.sigma_epsilon, obs.rho_eps)
    rcorr == BIG_NLL && return BIG_NLL
    val = 0.5 * (
        stats.K * 2.0 * log(p.sigma_epsilon) - obs.weights.log_weight_sum +
        ld_difference + lambda * obs.design.ydot - lambda^2 * quad - mean_corr + rcorr
    )
    return finite_or_big(val)
end

nll_value(solver::HutchSLQ, model, stats, params_full, obs, cache::Union{HutchCache,VSHutchCache}; seed) =
    nll_hutch_value(model, stats, solver, params_full, obs, cache; seed=seed)

# Final coefficient reconstruction uses exactly the fitted observation
# weighting, including eta-dependent AR(1) control products. Every network
# solve is iterative, and failures are explicit rather than silently falling
# back to Cholesky. mean_profile_correction consumes borrowed PCG results
# before the next solve and returns independently owned coefficient storage.
function final_mean_profile(solver::HutchSLQ, model, stats, obs, decoded,
    cache::Union{HutchCache,VSHutchCache})
    lambda = refresh_hutch_cache!(cache, obs, decoded)
    isfinite(lambda) && lambda > 0.0 || throw(MeanProfileError(
        "Final mean-profile residual precision is nonfinite or nonpositive."))
    solve_M = v -> hutch_mean_solve!(cache, solver, v)
    return mean_profile_correction(obs.mean_stats, lambda,
        obs.design.projected_y, solve_M, cache.mean::MeanProfileWorkspace;
        symmetry_rtol=max(1e-10, 10 * solver.cg_tol))
end

nelder_g_abstol(::HutchSLQ, g_rel::Float64) = g_rel

nelder_simplexer(solver::HutchSLQ) =
    AffineSimplexer(solver.simplex_shift, solver.simplex_scale)

# No gradient polish for the stochastic objective: finite differences of a
# noisy function are dominated by probe noise.
polish(::HutchSLQ, obj, res, verbose::Bool) = res, 0.0
