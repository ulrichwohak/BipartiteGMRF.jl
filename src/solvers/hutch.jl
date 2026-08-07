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
    )
end

make_nll_cache(solver::HutchSLQ, model::AbstractBipartiteModel, stats::BipartiteGMRFStats) =
    make_hutch_cache(model, stats, solver)

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

function nll_hutch_value(
    model::AbstractBipartiteModel,
    stats::BipartiteGMRFStats,
    solver::HutchSLQ,
    params_full::Vector{Float64},
    obs::ObservationStats,
    cache::Union{HutchCache,VSHutchCache};
    seed::Int,
)
    p = unpack_params(params_full; rho_limit=rho_limit(model))
    all(isfinite, (p.rho, p.sigma_a, p.sigma_z, p.sigma_epsilon)) || return BIG_NLL
    p.sigma_a > 0 && p.sigma_z > 0 && p.sigma_epsilon > 0 || return BIG_NLL

    lambda = 1.0 / p.sigma_epsilon^2
    set_q_params!(cache.qop, p.rho, p.sigma_a, p.sigma_z)
    cache.mop.lambda = lambda
    # VtV only changes across evaluations when rho_eps is re-estimated;
    # skip the O(n) diagonal extraction otherwise.
    if cache.mop.VtV !== obs.design.VtV
        cache.mop.VtV = obs.design.VtV
        cache.dV .= diag(obs.design.VtV)
    end
    Qdiag = q_diag(model, p.rho, p.sigma_a, p.sigma_z)
    @. cache.Mdiag = Qdiag + lambda * cache.dV

    x, ok, _, _ = pcg_solve!(cache.pcg, cache.mop, obs.design.projected_y;
        tol=solver.cg_tol, maxiter=solver.cg_maxiter, Mdiag=cache.Mdiag)
    ok || return BIG_NLL
    quad = dot(obs.design.projected_y, x)
    isfinite(quad) || return BIG_NLL

    mean_corr = 0.0
    if obs.mean_stats !== nothing
        ms = obs.mean_stats
        function pcg_solve_M(v)
            sol, ok_s, _, _ = pcg_solve!(cache.pcg, cache.mop, v;
                tol=solver.cg_tol, maxiter=solver.cg_maxiter, Mdiag=cache.Mdiag)
            ok_s || error("PCG failed for mean-structure solve")
            return sol
        end
        try
            mean_corr, _ = mean_profile_correction(ms, lambda,
                obs.design.projected_y, pcg_solve_M)
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

nelder_g_abstol(::HutchSLQ, g_rel::Float64) = g_rel

# No gradient polish for the stochastic objective: finite differences of a
# noisy function are dominated by probe noise.
polish(::HutchSLQ, obj, res, verbose::Bool) = res, 0.0
