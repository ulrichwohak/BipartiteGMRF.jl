# Fitted parameters rescaled to the standardized outcome units used
# internally by the objective and the precision matrices.
function scaled_params(result::GMRFResult)
    s = result.stats.y_std
    return (
        rho = result.rho,
        sigma_a = result.sigma_a / s,
        sigma_z = result.sigma_z / s,
        sigma_epsilon = result.sigma_epsilon / s,
    )
end

function target_weight_vector(stats::BipartiteGMRFStats, target::Symbol)
    t = normalize_decomp_target(target)
    T = Float64.(stats.decomp.T)
    if t == :estimation
        if stats.weighting.observations == :raw
            return T, :annual
        elseif stats.weighting.observations == :edge
            return ones(Float64, length(T)), :mean
        else
            return effective_match_weights(stats.decomp.T, stats.rho_eps_likelihood), :mean
        end
    elseif t == :personyear
        return T, :annual
    else
        return ones(Float64, length(T)), :mean
    end
end

function decomp_target_stats(stats::BipartiteGMRFStats, target::Symbol)
    t = normalize_decomp_target(target)
    weights, residual_level = target_weight_vector(stats, t)
    design = build_weighted_V_stats(
        stats.decomp.f,
        stats.decomp.w,
        stats.decomp.y,
        weights,
        stats.N_firms,
        stats.N_workers,
    )
    W = sum(weights)
    T = Float64.(stats.decomp.T)
    ydot_total = residual_level == :annual ? design.ydot + stats.personyear_within_ss : design.ydot
    observed_second_moment = stats.y_std^2 * ydot_total / W
    return (
        design = design,
        target = t,
        residual_level = residual_level,
        weights = weights,
        weight_sum = W,
        weight_over_T_sum = sum(weights ./ T),
        observed_second_moment = observed_second_moment,
    )
end

function residual_decomp_components(stats::BipartiteGMRFStats, sigma_epsilon_original::Float64, target_stats)
    sigma2 = sigma_epsilon_original^2
    mean_factor = target_stats.weight_over_T_sum / target_stats.weight_sum
    if stats.weighting.observations == :effective
        rho_eps = stats.rho_eps_likelihood
        V_eta = rho_eps * sigma2
        V_u_annual = (1.0 - rho_eps) * sigma2
        V_u_target = target_stats.residual_level == :annual ? V_u_annual : V_u_annual * mean_factor
        return (
            V_eta_match = V_eta,
            V_u_annual = V_u_annual,
            V_u_target = V_u_target,
            V_eps_target = V_eta + V_u_target,
            mean_residual_factor = mean_factor,
            model = :compound_symmetric,
        )
    end
    V_u_target = target_stats.residual_level == :annual ? sigma2 : sigma2 * mean_factor
    return (
        V_eta_match = 0.0,
        V_u_annual = sigma2,
        V_u_target = V_u_target,
        V_eps_target = V_u_target,
        mean_residual_factor = mean_factor,
        model = :iid,
    )
end

"""
Run `probes` Hutchinson probes through `solve_probe(v) -> (u, ok, relres)` and
accumulate the firm, worker, and cross quadratic-form estimates weighted by
the decomposition design. Returns the accumulators and the count of probes
that converged.
"""
function hutchinson_trace_blocks(
    solve_probe,
    design::DesignStats,
    N_firms::Int,
    n::Int,
    probes::Int,
    seed::Int,
    verbose::Bool,
    label::String,
)
    rng = MersenneTwister(seed)
    v = Vector{Float64}(undef, n)
    wv_f = Vector{Float64}(undef, N_firms)
    wv_w = Vector{Float64}(undef, n - N_firms)
    acc_f = 0.0
    acc_w = 0.0
    acc_cross = 0.0
    ok_count = 0

    for t in 1:probes
        @inbounds for i in 1:n
            v[i] = rand(rng, Bool) ? 1.0 : -1.0
        end
        u, ok, relres = solve_probe(v)
        if !ok
            verbose && @info "$(label) PCG did not converge" probe=t relres=relres
            continue
        end
        ok_count += 1
        vf = view(v, 1:N_firms)
        vw = view(v, N_firms + 1:n)
        uf = view(u, 1:N_firms)
        uw = view(u, N_firms + 1:n)
        mul!(wv_f, design.FF, uf)
        acc_f += dot(vf, wv_f)
        mul!(wv_w, design.WW, uw)
        acc_w += dot(vw, wv_w)
        mul!(wv_f, design.A_obs, uw)
        mul!(wv_w, design.At_obs, uf)
        acc_cross += dot(vf, wv_f) + dot(vw, wv_w)
    end
    return (f=acc_f, w=acc_w, cross=acc_cross, ok=ok_count)
end

function _decompose_model(
    result::GMRFResult;
    probes::Int=200,
    seed::Int=42,
    target::Symbol=result.stats.weighting.target,
    verbose::Bool=false,
)
    probes > 0 || throw(ArgumentError("probes must be positive."))
    model = result.model
    stats = result.stats
    p = scaled_params(result)
    n = stats.N_firms + stats.N_workers
    dstats = decomp_target_stats(stats, target)

    local solve_probe
    if result.solver isa ExactCholesky
        Q = model_precision(model, p.rho, p.sigma_a, p.sigma_z)
        FQ = cholesky(Symmetric(Q))
        solve_probe = v -> (FQ \ v, true, 0.0)
        method = :hutch_cholesky
    else
        qop = q_operator(model, p.rho, p.sigma_a, p.sigma_z)
        ws = PCGWorkspace(n)
        Qdiag = q_diag(model, p.rho, p.sigma_a, p.sigma_z)
        solve_probe = function (v)
            u, ok, _, relres = pcg_solve!(ws, qop, v;
                tol=result.solver.cg_tol, maxiter=result.solver.cg_maxiter, Mdiag=Qdiag)
            (u, ok, relres)
        end
        method = :hutch_pcg
    end

    acc = hutchinson_trace_blocks(
        solve_probe, dstats.design, stats.N_firms, n, probes, seed + 77_777, verbose,
        "model decomposition",
    )
    acc.ok > 0 || throw(ErrorException("All model decomposition probes failed."))

    scale = stats.y_std^2 / (Float64(acc.ok) * dstats.weight_sum)
    V_firm = acc.f * scale
    V_worker = acc.w * scale
    V_cross = acc.cross * scale
    resid = residual_decomp_components(stats, result.sigma_epsilon, dstats)
    V_epsilon = resid.V_eps_target
    V_total = V_firm + V_worker + V_cross + V_epsilon
    return VarianceDecomposition(
        V_firm,
        V_worker,
        V_cross,
        V_epsilon,
        V_total,
        probes,
        dstats.target,
        :model,
        method,
        method == :hutch_pcg ? acc.ok : nothing,
        (
            residual_model = resid.model,
            residual_level = dstats.residual_level,
            weight_sum = dstats.weight_sum,
            observed_second_moment = dstats.observed_second_moment,
            V_eta_match = resid.V_eta_match,
            V_u_annual = resid.V_u_annual,
            V_u_target = resid.V_u_target,
        ),
    )
end
