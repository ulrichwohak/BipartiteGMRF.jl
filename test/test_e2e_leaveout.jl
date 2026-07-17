using Graphs
using LeaveOut
using Printf

# Edge-weighted variance components from Σ = Q⁻¹
function edge_variance_components(Σ, firm_ids, worker_ids, n_firms)
    K = length(firm_ids)
    var_f = sum(Σ[f, f] for f in firm_ids) / K
    var_w = sum(Σ[n_firms + w, n_firms + w] for w in worker_ids) / K
    cov_fw = sum(Σ[f, n_firms + w] for (f, w) in zip(firm_ids, worker_ids)) / K
    corr = cov_fw / sqrt(var_f * var_w)
    return (; var_firm=var_f, var_worker=var_w, cov=cov_fw, corr=corr)
end

function print_comparison(label, var_firm, var_worker, cov, corr, var_eps)
    @printf("  %-12s  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f\n",
            label, var_firm, var_worker, cov, corr, var_eps)
end

@testset "E2E: MLE vs LeaveOut parameter recovery (10 reps)" begin
    n_reps = 10
    n_firms = 100
    n_workers = 150
    n_edges = 350
    truth = (rho=0.3, sigma_a=1.0, sigma_z=0.5, sigma_epsilon=1.0)

    # Fixed graph across reps (seed=42 for graph generation)
    graph_rng = MersenneTwister(42)
    edges = connected_random_edges(graph_rng, n_firms, n_workers, n_edges)
    I_idx = [e[1] for e in edges]
    J_idx = [e[2] for e in edges]
    A = sparse(I_idx, J_idx, ones(length(edges)), n_firms, n_workers)

    # Resolve VS feasibility once
    vs_result = BipartiteGMRF.prepare_vs_feasibility(VarianceStablePrior(rho_limit=:auto), A)
    resolved_limit = vs_result.metadata.resolved_rho_limit
    @test resolved_limit > truth.rho

    model = BipartiteGMRF.to_model(VarianceStablePrior(rho_limit=resolved_limit), A)

    # Model-implied variance components at TRUE parameters (same across reps)
    Q_true = BipartiteGMRF.model_precision(model, truth.rho, truth.sigma_a, truth.sigma_z)
    Σ_true = inv(Matrix(Q_true))
    vc_true = edge_variance_components(Σ_true, I_idx, J_idx, n_firms)

    # LeaveOut design (same across reps)
    g = SimpleGraph(n_firms + n_workers)
    for (f, w) in edges
        add_edge!(g, f, n_firms + w)
    end
    d = Design(g)
    block, ekeep, nodemap = largest_block(d)
    managers_in_block = [i for (i, orig) in enumerate(nodemap) if orig <= n_firms]
    pb = prepare(block)

    # Accumulators (var_firm, var_worker, cov, corr, var_eps)
    mle_acc = zeros(5)
    kss_acc = zeros(5)

    for rep in 1:n_reps
        sim_rng = MersenneTwister(100 + rep)
        sim = simulate(model, I_idx, J_idx;
            ρ=truth.rho, σ_a=truth.sigma_a, σ_z=truth.sigma_z, σ_ε=truth.sigma_epsilon,
            rng=sim_rng)

        # MLE
        df = DataFrame(firm_id=sim.firm_ids, worker_id=sim.worker_ids .+ 1000, y=sim.y)
        mle_result = gmrf_mle(df;
            prior=VarianceStablePrior(rho_limit=:auto),
            solver=ExactCholesky(optim_iters=200, polish=true),
            standardize=false, decompose=false, seed=rep, verbose=false)

        Q_mle = BipartiteGMRF.model_precision(model, mle_result.rho, mle_result.sigma_a, mle_result.sigma_z)
        Σ_mle = inv(Matrix(Q_mle))
        vc_mle = edge_variance_components(Σ_mle, I_idx, J_idx, n_firms)
        mle_acc .+= [vc_mle.var_firm, vc_mle.var_worker, vc_mle.cov, vc_mle.corr, mle_result.sigma_epsilon^2]

        # KSS
        edge_to_y = Dict((f, n_firms + w) => sim.y[k] for (k, (f, w)) in enumerate(edges))
        Y_full = [edge_to_y[e] for e in d.edges]
        lo = LeaveOut.decompose(pb, Y_full[ekeep]; managers=managers_in_block)
        kss_acc .+= [lo.var_theta, lo.var_psi, lo.cov, lo.corr, lo.var_eps]
    end

    mle_avg = mle_acc ./ n_reps
    kss_avg = kss_acc ./ n_reps

    # --- Report ---
    println()
    @printf("  %-12s  %8s  %8s  %8s  %8s  %8s\n",
            "", "Var(α)", "Var(z)", "Cov", "Corr", "Var(ε)")
    @printf("  %s\n", "-"^60)
    print_comparison("GMRF(true)", vc_true.var_firm, vc_true.var_worker, vc_true.cov, vc_true.corr, truth.sigma_epsilon^2)
    print_comparison("GMRF(MLE)",  mle_avg[1], mle_avg[2], mle_avg[3], mle_avg[4], mle_avg[5])
    print_comparison("KSS",        kss_avg[1], kss_avg[2], kss_avg[3], kss_avg[4], kss_avg[5])
    @printf("  %s\n", "-"^60)
    @printf("  True params: ρ=%.4f  σ_a=%.4f  σ_z=%.4f  σ_ε=%.4f\n",
            truth.rho, truth.sigma_a, truth.sigma_z, truth.sigma_epsilon)
    @printf("  Edges: %d total, %d after pruning, %d reps\n", length(edges), length(ekeep), n_reps)
    println()

    # --- Assertions (on averages) ---
    @test abs(mle_avg[1] - vc_true.var_firm) < 0.3
    @test abs(mle_avg[2] - vc_true.var_worker) < 0.15
    @test abs(mle_avg[5] - truth.sigma_epsilon^2) < 0.15

    @test abs(kss_avg[1] - vc_true.var_firm) < 0.5
    @test abs(kss_avg[2] - vc_true.var_worker) < 0.15
    @test abs(kss_avg[5] - truth.sigma_epsilon^2) < 0.3
end
