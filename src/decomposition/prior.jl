function target_weight_vector(problem::GMRFProblem, target::Symbol)
    t = normalize_decomp_target(target)
    T = Float64.(problem.decomp_T)
    if t == :estimation
        if problem.weighting.observations == :raw
            return T, :annual
        elseif problem.weighting.observations == :edge
            return ones(Float64, length(T)), :mean
        else
            return effective_match_weights(problem.decomp_T, problem.rho_eps_likelihood), :mean
        end
    elseif t == :personyear
        return T, :annual
    else
        return ones(Float64, length(T)), :mean
    end
end

function decomp_target_stats(problem::GMRFProblem, target::Symbol)
    t = normalize_decomp_target(target)
    weights, residual_level = target_weight_vector(problem, t)
    stats = build_weighted_V_stats(
        problem.decomp_f_rows,
        problem.decomp_w_cols,
        problem.decomp_y,
        weights,
        problem.N_firms,
        problem.N_workers,
    )
    W = sum(weights)
    T = Float64.(problem.decomp_T)
    ydot_total = residual_level == :annual ? stats.ydot + problem.personyear_within_ss : stats.ydot
    observed_second_moment = problem.y_std^2 * ydot_total / W
    return merge(stats, (
        target = t,
        residual_level = residual_level,
        weights = weights,
        weight_sum = W,
        weight_over_T_sum = sum(weights ./ T),
        observed_second_moment = observed_second_moment,
    ))
end

function residual_decomp_components(problem::GMRFProblem, sigma_epsilon_original::Float64, target_stats)
    sigma2 = sigma_epsilon_original^2
    mean_factor = target_stats.weight_over_T_sum / target_stats.weight_sum
    if problem.weighting.observations == :effective
        rho_eps = problem.rho_eps_likelihood
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
    prior_decomposition(result::GMRFResult; probes=200, seed=42,
                        target=result.problem.weighting.target,
                        verbose=false)

Estimate the prior variance decomposition implied by a fitted model.

Returns a `VarianceDecomposition` containing firm, worker, cross, residual, and
total variance components for the requested target. `probes` controls the
Hutchinson trace estimator used for latent-field variance terms.
"""
function prior_decomposition(
    result::GMRFResult;
    probes::Int=200,
    seed::Int=42,
    target::Symbol=result.problem.weighting.target,
    verbose::Bool=false,
)
    probes > 0 || throw(ArgumentError("probes must be positive."))
    problem = result.problem
    p = unpack_params(result.theta_unconstrained)
    sigma_a = result.sigma_a / problem.y_std
    sigma_z = result.sigma_z / problem.y_std
    sigma_e = result.sigma_epsilon / problem.y_std
    n = problem.N_firms + problem.N_workers
    dstats = decomp_target_stats(problem, target)
    cnt_f, cnt_w = dstats.cnt_f, dstats.cnt_w
    acc_f = 0.0
    acc_w = 0.0
    acc_cross = 0.0
    ok_count = 0
    rng = MersenneTwister(seed + 77_777)
    v = Vector{Float64}(undef, n)
    wv_f = Vector{Float64}(undef, problem.N_firms)
    wv_w = Vector{Float64}(undef, problem.N_workers)

    if result.solver isa ExactCholesky
        Q = precision_matrix(problem, p.rho, sigma_a, sigma_z)
        FQ = cholesky(Symmetric(Q))
        for _ in 1:probes
            @inbounds for i in 1:n
                v[i] = rand(rng, Bool) ? 1.0 : -1.0
            end
            u = FQ \ v
            vf = view(v, 1:problem.N_firms)
            vw = view(v, problem.N_firms + 1:n)
            uf = view(u, 1:problem.N_firms)
            uw = view(u, problem.N_firms + 1:n)
            @inbounds for i in 1:problem.N_firms
                acc_f += cnt_f[i] * vf[i] * uf[i]
            end
            @inbounds for j in 1:problem.N_workers
                acc_w += cnt_w[j] * vw[j] * uw[j]
            end
            mul!(wv_f, dstats.A_obs, uw)
            mul!(wv_w, dstats.At_obs, uf)
            acc_cross += dot(vf, wv_f) + dot(vw, wv_w)
        end
        ok_count = probes
        method = :hutch_cholesky
        pcg_count = nothing
    else
        qop = q_operator(problem, p.rho, sigma_a, sigma_z)
        ws = PCGWorkspace(n)
        Qdiag = q_diag(problem, p.rho, sigma_a, sigma_z)
        for t in 1:probes
            @inbounds for i in 1:n
                v[i] = rand(rng, Bool) ? 1.0 : -1.0
            end
            u, ok, _, relres = pcg_solve!(ws, qop, v;
                tol=result.solver.cg_tol, maxiter=result.solver.cg_maxiter, Mdiag=Qdiag)
            if ok
                ok_count += 1
            elseif verbose
                @info "prior decomposition PCG did not converge" probe=t relres=relres
                continue
            end
            vf = view(v, 1:problem.N_firms)
            vw = view(v, problem.N_firms + 1:n)
            uf = view(u, 1:problem.N_firms)
            uw = view(u, problem.N_firms + 1:n)
            @inbounds for i in 1:problem.N_firms
                acc_f += cnt_f[i] * vf[i] * uf[i]
            end
            @inbounds for j in 1:problem.N_workers
                acc_w += cnt_w[j] * vw[j] * uw[j]
            end
            mul!(wv_f, dstats.A_obs, uw)
            mul!(wv_w, dstats.At_obs, uf)
            acc_cross += dot(vf, wv_f) + dot(vw, wv_w)
        end
        method = :hutch_pcg
        pcg_count = ok_count
    end

    ok_count > 0 || throw(ErrorException("All prior decomposition probes failed."))
    scale = problem.y_std^2 / (Float64(ok_count) * dstats.weight_sum)
    V_firm = acc_f * scale
    V_worker = acc_w * scale
    V_cross = acc_cross * scale
    resid = residual_decomp_components(problem, sigma_e * problem.y_std, dstats)
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
        :prior,
        method,
        pcg_count,
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
