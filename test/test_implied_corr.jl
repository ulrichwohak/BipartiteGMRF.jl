function implied_problem(edges)
    return GMRFProblem(
        DataFrame(
            firm_id=first.(edges),
            worker_id=last.(edges),
            y=collect(1.0:length(edges)),
        );
        prior=VarianceStablePrior(),
        standardize=false,
    )
end

@testset "implied correlations" begin
    rho = 0.4

    path_problem = implied_problem([(1, 1), (2, 1), (2, 2)])
    path_result = implied_correlations(path_problem, rho; selection=:all, batch_size=2)
    @test path_result isa ImpliedCorrResult
    @test all(edge -> isapprox(edge.correlation, rho; atol=1e-12), path_result.edges)
    @test all(node -> isapprox(node.variance_inflation, 1.0; atol=1e-12), path_result.firms)
    @test all(node -> isapprox(node.variance_inflation, 1.0; atol=1e-12), path_result.workers)
    @test all(edge -> edge.stratum == :tree, path_result.edges)
    @test implied_corr_functional(path_result).value ≈ rho atol=1e-12
    @test implied_corr_functional(path_result).method == :exact

    cycle_edges = [(firm, worker) for firm in 1:2 for worker in 1:2]
    cycle_problem = @test_warn "contains a cycle" implied_problem(cycle_edges)
    cycle_result = implied_correlations(cycle_problem, rho; selection=:all, batch_size=3)
    expected_corr = (rho + rho^3) / (1 + rho^4)
    expected_inflation = (1 + rho^4) / (1 - rho^4)
    @test all(
        edge -> isapprox(edge.correlation, expected_corr; atol=1e-12),
        cycle_result.edges,
    )
    @test all(
        node -> isapprox(node.variance_inflation, expected_inflation; atol=1e-12),
        vcat(cycle_result.firms, cycle_result.workers),
    )
    @test all(edge -> edge.stratum == :core, cycle_result.edges)
    @test any(edge -> edge.hotspot, cycle_result.edges)

    dense_edges = vcat(
        [(firm, worker) for firm in 1:3 for worker in 1:3],
        [(4, 3), (4, 4)],
    )
    dense_problem = @test_warn "contains a cycle" implied_problem(dense_edges)
    dense_result = implied_correlations(dense_problem, 0.3; selection=:all, batch_size=2)
    B = BipartiteGMRF.precision_matrix(dense_problem, 0.3, 1.0, 1.0)
    B_inverse = inv(Matrix(B))
    for edge in dense_result.edges
        firm = edge.firm_index
        worker = dense_problem.N_firms + edge.worker_index
        reference = B_inverse[firm, worker] / sqrt(B_inverse[firm, firm] * B_inverse[worker, worker])
        @test edge.correlation ≈ reference atol=1e-12
    end
    for node in dense_result.firms
        @test node.b_inverse_diagonal ≈ B_inverse[node.index, node.index] atol=1e-12
    end
    for node in dense_result.workers
        global_index = dense_problem.N_firms + node.index
        @test node.b_inverse_diagonal ≈ B_inverse[global_index, global_index] atol=1e-12
    end

    batch_one = implied_correlations(dense_problem, 0.3; selection=:all, batch_size=1)
    batch_all = implied_correlations(dense_problem, 0.3; selection=:all, batch_size=20)
    @test isapprox(
        getfield.(batch_one.edges, :correlation),
        getfield.(batch_all.edges, :correlation);
        atol=1e-12,
    )

    sampled_edges = vcat(
        cycle_edges,
        [(firm, isodd(firm) ? 1 : 2) for firm in 3:8],
        [(firm, firm) for firm in 3:8],
    )
    sampled_problem = @test_warn "contains a cycle" implied_problem(sampled_edges)
    sampled_first = implied_correlations(
        sampled_problem,
        0.3;
        selection=:stratified,
        outside_sample=2,
        seed=2026,
    )
    sampled_second = implied_correlations(
        sampled_problem,
        0.3;
        selection=:stratified,
        outside_sample=2,
        seed=2026,
    )
    sampled_keys(result) = [(edge.firm_id, edge.worker_id) for edge in result.edges]
    @test sampled_keys(sampled_first) == sampled_keys(sampled_second)
    @test sampled_first.selected_counts == (core=4, adjacent=2, tree=2)
    @test sampled_first.stratum_counts == (core=4, adjacent=6, tree=6)
    @test sampled_first.functional_method == :stratified
    @test implied_corr_functional(sampled_first).method == :stratified

    complete_edges = [(firm, worker) for firm in 1:3 for worker in 1:3]
    infeasible_problem = @test_warn "contains a cycle" implied_problem(complete_edges)
    @test_throws ArgumentError implied_correlations(infeasible_problem, 0.5; selection=:all)
    @test_throws ArgumentError implied_correlations(path_problem, 1.0; selection=:all)
    @test_throws ArgumentError implied_correlations(
        GMRFProblem(tree_df(n=3); standardize=false),
        0.2,
    )
    @test_throws ArgumentError implied_correlations(path_problem, rho; selection=:invalid)
end
