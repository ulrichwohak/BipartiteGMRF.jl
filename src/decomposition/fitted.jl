function _decompose_fitted(
    result::GMRFResult;
    probes::Int=200,
    seed::Int=42,
    target::Symbol=result.stats.weighting.target,
    verbose::Bool=false,
)
    probes > 0 || throw(ArgumentError("probes must be positive."))
    model = result.model
    stats = result.stats
    p = unpack_params(result.theta_unconstrained; rho_limit=rho_limit(model))
    sigma_a = result.sigma_a / stats.y_std
    sigma_z = result.sigma_z / stats.y_std
    sigma_e = result.sigma_epsilon / stats.y_std
    lambda = 1.0 / sigma_e^2
    n = stats.N_firms + stats.N_workers
    dstats = decomp_target_stats(stats, target)
    cnt_f, cnt_w = dstats.cnt_f, dstats.cnt_w

    rhs = lambda .* stats.projected_y
    local solve_probe
    if result.solver isa ExactCholesky
        M = fitted_precision(model, stats.VtV, p.rho, sigma_a, sigma_z, sigma_e)
        FM = cholesky(Symmetric(M))
        theta_hat = FM \ rhs
        solve_probe = v -> (FM \ v, true, 0.0)
        method = :hutch_cholesky
    else
        qop = q_operator(model, p.rho, sigma_a, sigma_z)
        mop = MOp(qop, stats.VtV, zeros(n), lambda)
        ws = PCGWorkspace(n)
        Mdiag = q_diag(model, p.rho, sigma_a, sigma_z) .+ lambda .* Vector{Float64}(diag(stats.VtV))
        theta_ref, ok, _, relres = pcg_solve!(ws, mop, rhs;
            tol=result.solver.cg_tol, maxiter=result.solver.cg_maxiter, Mdiag=Mdiag)
        ok || throw(ErrorException("Posterior mode PCG did not converge; relres=$(relres)."))
        theta_hat = copy(theta_ref)
        solve_probe = function (v)
            u, okp, _, rr = pcg_solve!(ws, mop, v;
                tol=result.solver.cg_tol, maxiter=result.solver.cg_maxiter, Mdiag=Mdiag)
            (u, okp, rr)
        end
        method = :hutch_pcg
    end

    th_f = view(theta_hat, 1:stats.N_firms)
    th_w = view(theta_hat, stats.N_firms + 1:n)
    tmp = Vector{Float64}(undef, stats.N_firms)
    mul!(tmp, dstats.A_obs, th_w)
    qa = dot(cnt_f, th_f .^ 2)
    qz = dot(cnt_w, th_w .^ 2)
    qaz = dot(th_f, tmp)

    rng = MersenneTwister(seed + 88_888)
    v = Vector{Float64}(undef, n)
    wv_f = Vector{Float64}(undef, stats.N_firms)
    wv_w = Vector{Float64}(undef, stats.N_workers)
    acc_f = 0.0
    acc_w = 0.0
    acc_cross = 0.0
    ok_count = 0

    for t in 1:probes
        @inbounds for i in 1:n
            v[i] = rand(rng, Bool) ? 1.0 : -1.0
        end
        u, ok, relres = solve_probe(v)
        if ok
            ok_count += 1
        elseif verbose
            @info "fitted decomposition PCG did not converge" probe=t relres=relres
            continue
        end
        vf = view(v, 1:stats.N_firms)
        vw = view(v, stats.N_firms + 1:n)
        uf = view(u, 1:stats.N_firms)
        uw = view(u, stats.N_firms + 1:n)
        @inbounds for i in 1:stats.N_firms
            acc_f += cnt_f[i] * vf[i] * uf[i]
        end
        @inbounds for j in 1:stats.N_workers
            acc_w += cnt_w[j] * vw[j] * uw[j]
        end
        mul!(wv_f, dstats.A_obs, uw)
        mul!(wv_w, dstats.At_obs, uf)
        acc_cross += dot(vf, wv_f) + dot(vw, wv_w)
    end

    ok_count > 0 || throw(ErrorException("All fitted decomposition probes failed."))
    quad_scale = stats.y_std^2 / dstats.weight_sum
    trace_scale = stats.y_std^2 / (Float64(ok_count) * dstats.weight_sum)
    V_firm = qa * quad_scale + acc_f * trace_scale
    V_worker = qz * quad_scale + acc_w * trace_scale
    cov_cross = qaz * quad_scale + acc_cross * trace_scale / 2.0
    V_cross = 2.0 * cov_cross
    resid = residual_decomp_components(stats, sigma_e * stats.y_std, dstats)
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
        :fitted,
        method,
        result.solver isa ExactCholesky ? nothing : ok_count,
        (
            covariance = cov_cross,
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
