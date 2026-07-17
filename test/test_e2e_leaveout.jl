using Graphs
using LeaveOut

@testset "E2E: MLE vs LeaveOut parameter recovery" begin
    seed = 42
    rng = MersenneTwister(seed)
    n_firms = 100
    n_workers = 150
    n_edges = 350
    truth = (rho=0.3, sigma_a=1.0, sigma_z=0.5, sigma_epsilon=1.0)

    # 1. Bipartite Erdős-Rényi graph (reuse fixture)
    edges = connected_random_edges(rng, n_firms, n_workers, n_edges)
    I_idx = [e[1] for e in edges]
    J_idx = [e[2] for e in edges]
    A = sparse(I_idx, J_idx, ones(length(edges)), n_firms, n_workers)

    # 2. Resolve VS feasibility to get numeric rho_limit, then build model + simulate
    vs_result = BipartiteGMRF.prepare_vs_feasibility(VarianceStablePrior(rho_limit=:auto), A)
    resolved_limit = vs_result.metadata.resolved_rho_limit
    @test resolved_limit > truth.rho  # ρ = 0.5 must be feasible

    model = BipartiteGMRF.to_model(VarianceStablePrior(rho_limit=resolved_limit), A)
    sim = simulate(model, I_idx, J_idx;
        ρ=truth.rho, σ_a=truth.sigma_a, σ_z=truth.sigma_z, σ_ε=truth.sigma_epsilon,
        rng=MersenneTwister(seed + 1))

    # 3. MLE recovery (offset worker IDs to avoid namespace collision)
    df = DataFrame(firm_id=sim.firm_ids, worker_id=sim.worker_ids .+ 1000, y=sim.y)
    mle_result = gmrf_mle(df;
        prior=VarianceStablePrior(rho_limit=:auto),
        solver=ExactCholesky(optim_iters=200, polish=true),
        standardize=false, decompose=false, seed=seed, verbose=false)

    @info "MLE results" rho=mle_result.rho sigma_a=mle_result.sigma_a sigma_z=mle_result.sigma_z sigma_eps=mle_result.sigma_epsilon
    @info "Truth"       rho=truth.rho sigma_a=truth.sigma_a sigma_z=truth.sigma_z sigma_eps=truth.sigma_epsilon

    @test abs(mle_result.rho - truth.rho) < 0.15
    @test abs(log(mle_result.sigma_a / truth.sigma_a)) < 0.25
    @test abs(log(mle_result.sigma_z / truth.sigma_z)) < 0.25
    @test abs(log(mle_result.sigma_epsilon / truth.sigma_epsilon)) < 0.20

    # 4. LeaveOut estimation
    # Nodes 1:n_firms = firms, (n_firms+1):(n_firms+n_workers) = workers
    g = SimpleGraph(n_firms + n_workers)
    for (f, w) in edges
        add_edge!(g, f, n_firms + w)
    end

    d = Design(g)

    # Align Y to Design's canonical edge order
    edge_to_y = Dict((f, n_firms + w) => sim.y[k] for (k, (f, w)) in enumerate(edges))
    Y_full = [edge_to_y[e] for e in d.edges]

    block, ekeep, nodemap = largest_block(d)
    @test length(ekeep) > 0.8 * length(edges)

    # Identify which renumbered nodes are firms (critical after relabeling)
    managers_in_block = [i for (i, orig) in enumerate(nodemap) if orig <= n_firms]

    pb = prepare(block)
    lo = LeaveOut.decompose(pb, Y_full[ekeep]; managers=managers_in_block)

    @info "LeaveOut results" var_firm=lo.var_theta var_worker=lo.var_psi cov=lo.cov corr=lo.corr var_eps=lo.var_eps
    @info "Truth (variances)" var_firm=truth.sigma_a^2 var_worker=truth.sigma_z^2 var_eps=truth.sigma_epsilon^2
    @info "Edges" total=length(edges) after_pruning=length(ekeep)

    # LeaveOut variance estimates vs true population variances
    # VarianceStable: Var(α_i) = σ_a², Var(z_j) = σ_z² (exact on forests, approximate on cyclic graphs)
    @test abs(lo.var_theta - truth.sigma_a^2) < 0.5
    @test abs(lo.var_psi - truth.sigma_z^2) < 0.3
    @test abs(lo.var_eps - truth.sigma_epsilon^2) < 0.5
end
