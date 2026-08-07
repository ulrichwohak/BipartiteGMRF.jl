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
    p = scaled_params(result)
    lambda = 1.0 / p.sigma_epsilon^2
    n = stats.N_firms + stats.N_workers
    dstats = decomp_target_stats(stats, target)

    rhs = lambda .* stats.design.projected_y
    local solve_probe
    if result.solver isa ExactCholesky
        M = fitted_precision(model, stats.design.VtV, p.rho, p.sigma_a, p.sigma_z, p.sigma_epsilon)
        FM = cholesky(Symmetric(M))
        theta_hat = FM \ rhs
        solve_probe = v -> (FM \ v, true, 0.0)
        method = :hutch_cholesky
    else
        qop = q_operator(model, p.rho, p.sigma_a, p.sigma_z)
        mop = MOp(qop, stats.design.VtV, zeros(n), lambda)
        ws = PCGWorkspace(n)
        Mdiag = q_diag(model, p.rho, p.sigma_a, p.sigma_z) .+ lambda .* Vector{Float64}(diag(stats.design.VtV))
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

    design = dstats.design
    th_f = view(theta_hat, 1:stats.N_firms)
    th_w = view(theta_hat, stats.N_firms + 1:n)
    tmp_f = Vector{Float64}(undef, stats.N_firms)
    mul!(tmp_f, design.FF, th_f)
    qa = dot(th_f, tmp_f)
    mul!(tmp_f, design.A_obs, th_w)
    qaz = dot(th_f, tmp_f)
    tmp_w = Vector{Float64}(undef, n - stats.N_firms)
    mul!(tmp_w, design.WW, th_w)
    qz = dot(th_w, tmp_w)

    acc = hutchinson_trace_blocks(
        solve_probe, design, stats.N_firms, n, probes, seed + 88_888, verbose,
        "fitted decomposition",
    )
    acc.ok > 0 || throw(ErrorException("All fitted decomposition probes failed."))

    quad_scale = stats.y_std^2 / dstats.weight_sum
    trace_scale = stats.y_std^2 / (Float64(acc.ok) * dstats.weight_sum)
    V_firm = qa * quad_scale + acc.f * trace_scale
    V_worker = qz * quad_scale + acc.w * trace_scale
    cov_cross = qaz * quad_scale + acc.cross * trace_scale / 2.0
    V_cross = 2.0 * cov_cross
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
        :fitted,
        method,
        method == :hutch_pcg ? acc.ok : nothing,
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
