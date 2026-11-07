function nb_prune_df(edges; duplicate_edge=nothing)
    firm = first.(edges)
    worker = last.(edges)
    if duplicate_edge !== nothing
        duplicate_index = findfirst(==(duplicate_edge), edges)
        append!(firm, fill(first(duplicate_edge), 2))
        append!(worker, fill(last(duplicate_edge), 2))
        @assert duplicate_index !== nothing
    end
    return DataFrame(
        firm_id=firm,
        worker_id=worker,
        y=collect(1.0:length(firm)),
    )
end

@testset "connectivity-preserving NB edge pruning" begin
    forest = tree_df(n=6)
    forest_result = nb_prune_edges(forest; target=1.0, batch_size=1)
    @test forest_result isa NBPruneResult
    @test forest_result.lambda_before == 0.0
    @test forest_result.lambda_after == 0.0
    @test forest_result.target_met
    @test all(forest_result.keep)
    @test isempty(forest_result.dropped_edges)
    @test isempty(forest_result.audit)
    @test pruned_dataframe(forest, forest_result) == forest

    core_edges = [(firm, worker) for firm in 1:3 for worker in 1:3]
    tree_plus_core = vcat(core_edges, [(4, 3), (4, 4)])
    connected = nb_prune_df(tree_plus_core; duplicate_edge=(1, 1))
    connected_result = nb_prune_edges(
        connected;
        target=1.9,
        batch_size=1,
        step_fraction=0.05,
        seed=91,
    )
    @test connected_result.lambda_before ≈ 2.0 atol=1e-10
    @test connected_result.lambda_after < 1.9
    @test connected_result.target_met
    @test connected_result.components_before == 1
    @test connected_result.components_after == 1
    @test connected_result.protected_edges == 7
    @test !isempty(connected_result.audit)
    @test all(round -> round.lambda_after <= round.lambda_before + 1e-10, connected_result.audit)

    connected_pruned = pruned_dataframe(connected, connected_result)
    @test Set(connected_pruned.firm_id) == Set(connected.firm_id)
    @test Set(connected_pruned.worker_id) == Set(connected.worker_id)
    for group in groupby(connected, [:firm_id, :worker_id])
        rows = parentindices(group)[1]
        @test length(unique(connected_result.keep[rows])) == 1
    end
    dropped_unique = count(group -> !connected_result.keep[first(parentindices(group)[1])],
        groupby(connected, [:firm_id, :worker_id]))
    @test length(connected_result.dropped_edges) == dropped_unique

    repeated_result = nb_prune_edges(
        connected;
        target=1.9,
        batch_size=1,
        step_fraction=0.05,
        seed=91,
    )
    @test repeated_result.keep == connected_result.keep
    @test repeated_result.dropped_edges == connected_result.dropped_edges
    @test repeated_result.audit == connected_result.audit

    disconnected_edges = vcat(
        [(firm, worker) for firm in 1:4 for worker in 1:4],
        [(firm, worker) for firm in 5:8 for worker in 5:8],
    )
    disconnected = nb_prune_df(disconnected_edges)
    disconnected_result = nb_prune_edges(
        disconnected;
        target=2.8,
        batch_size=1,
        step_fraction=0.05,
        seed=92,
    )
    @test disconnected_result.lambda_before ≈ 3.0 atol=1e-10
    @test disconnected_result.lambda_after < 2.8
    @test disconnected_result.components_before == 2
    @test disconnected_result.components_after == 2
    @test any(edge -> edge.firm_id <= 4, disconnected_result.dropped_edges)
    @test any(edge -> edge.firm_id >= 5, disconnected_result.dropped_edges)

    auto_problem = GMRFProblem(
        connected_pruned;
        prior=VarianceStablePrior(rho_limit=:auto),
        standardize=false,
    )
    report = feasibility(auto_problem)
    @test report.safe
    @test report.lambda_nb ≈ connected_result.lambda_after atol=1e-10

    @test_throws ArgumentError nb_prune_edges(connected; target=0.0)
    @test_throws ArgumentError nb_prune_edges(connected; target=1.0, batch_size=0)
    @test_throws DimensionMismatch pruned_dataframe(connected[1:end-1, :], connected_result)
end
