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
    n_firms = 200
    n_workers = 300
    n_edges = 1000
    truth = (rho=0.2, sigma_a=1.0, sigma_z=0.5, sigma_epsilon=1.0)

    # --- Build graph and prune to bridgeless core ---
    graph_rng = MersenneTwister(42)
    raw_edges = connected_random_edges(graph_rng, n_firms, n_workers, n_edges)

    # Resolve NB spectral radius on full graph (conservative bound)
    A_full = sparse([e[1] for e in raw_edges], [e[2] for e in raw_edges],
                    ones(length(raw_edges)), n_firms, n_workers)
    vs_result = BipartiteGMRF.prepare_vs_feasibility(VarianceStablePrior(rho_limit=:auto), A_full)
    resolved_limit = vs_result.metadata.resolved_rho_limit
    @test resolved_limit > truth.rho

    # Build SimpleGraph from bipartite adjacency block matrix [0 A; A' 0]
    Z1 = spzeros(n_firms, n_firms)
    Z2 = spzeros(n_workers, n_workers)
    adj = [Z1 A_full; A_full' Z2]
    g_full = SimpleGraph(adj)
    d_full = Design(g_full)
    block, ekeep, nodemap = largest_block(d_full)

    # Map pruned graph back to bipartite adjacency
    # nodemap[i] = original node id; firms: orig ≤ n_firms, workers: orig > n_firms
    firm_orig = [nodemap[i] for i in 1:length(nodemap) if nodemap[i] <= n_firms]
    worker_orig = [nodemap[i] - n_firms for i in 1:length(nodemap) if nodemap[i] > n_firms]
    firm_reindex = Dict(orig => idx for (idx, orig) in enumerate(firm_orig))
    worker_reindex = Dict(orig => idx for (idx, orig) in enumerate(worker_orig))
    nf = length(firm_orig)
    nw = length(worker_orig)

    # Pruned edges in bipartite (firm_idx, worker_idx) coordinates
    pruned_edges = Tuple{Int,Int}[]
    for e in block.edges
        src_orig, dst_orig = nodemap[e[1]], nodemap[e[2]]
        if src_orig <= n_firms
            push!(pruned_edges, (firm_reindex[src_orig], worker_reindex[dst_orig - n_firms]))
        else
            push!(pruned_edges, (firm_reindex[dst_orig], worker_reindex[src_orig - n_firms]))
        end
    end
    I_idx = [e[1] for e in pruned_edges]
    J_idx = [e[2] for e in pruned_edges]
    A = sparse(I_idx, J_idx, ones(length(pruned_edges)), nf, nw)

    # Build model on pruned graph with resolved rho_limit
    model = BipartiteGMRF.to_model(VarianceStablePrior(rho_limit=resolved_limit), A)

    # Model-implied variance components at TRUE parameters
    Q_true = BipartiteGMRF.model_precision(model, truth.rho, truth.sigma_a, truth.sigma_z)
    Σ_true = inv(Matrix(Q_true))
    vc_true = edge_variance_components(Σ_true, I_idx, J_idx, nf)

    # LeaveOut design on pruned graph (no further pruning needed)
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

        # MLE on same pruned graph
        df = DataFrame(firm_id=sim.firm_ids, worker_id=sim.worker_ids .+ 10000, y=sim.y)
        mle_result = gmrf_mle(df;
            prior=VarianceStablePrior(rho_limit=resolved_limit),
            solver=ExactCholesky(optim_iters=200, polish=true),
            standardize=false, decompose=false, seed=rep, verbose=false)

        Q_mle = BipartiteGMRF.model_precision(model, mle_result.rho, mle_result.sigma_a, mle_result.sigma_z)
        Σ_mle = inv(Matrix(Q_mle))
        vc_mle = edge_variance_components(Σ_mle, I_idx, J_idx, nf)
        mle_acc .+= [vc_mle.var_firm, vc_mle.var_worker, vc_mle.cov, vc_mle.corr, mle_result.sigma_epsilon^2]

        # KSS on same pruned graph — Y aligned to block.edges
        edge_to_y = Dict(pruned_edges[k] => sim.y[k] for k in eachindex(pruned_edges))
        Y_block = Float64[]
        for e in block.edges
            src_orig, dst_orig = nodemap[e[1]], nodemap[e[2]]
            if src_orig <= n_firms
                push!(Y_block, edge_to_y[(firm_reindex[src_orig], worker_reindex[dst_orig - n_firms])])
            else
                push!(Y_block, edge_to_y[(firm_reindex[dst_orig], worker_reindex[src_orig - n_firms])])
            end
        end
        lo = LeaveOut.decompose(pb, Y_block; managers=managers_in_block)
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
    @printf("  Graph: %d firms, %d workers, %d edges (from %d raw, %d after pruning), %d reps\n",
            nf, nw, length(pruned_edges), length(raw_edges), length(ekeep), n_reps)
    println()

    # --- Assertions (on averages) ---
    @test abs(mle_avg[1] - vc_true.var_firm) < 0.3
    @test abs(mle_avg[2] - vc_true.var_worker) < 0.15
    @test abs(mle_avg[5] - truth.sigma_epsilon^2) < 0.15

    @test abs(kss_avg[1] - vc_true.var_firm) < 0.5
    @test abs(kss_avg[2] - vc_true.var_worker) < 0.15
    @test abs(kss_avg[5] - truth.sigma_epsilon^2) < 0.3
end
