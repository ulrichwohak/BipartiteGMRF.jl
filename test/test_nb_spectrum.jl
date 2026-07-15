function nb_test_matrix(n_firms, n_workers, edges)
    return sparse(
        first.(edges),
        last.(edges),
        ones(Float64, length(edges)),
        n_firms,
        n_workers,
    )
end

complete_bipartite_edges(n_firms, n_workers; firm_offset=0, worker_offset=0) = [
    (firm_offset + firm, worker_offset + worker)
    for firm in 1:n_firms for worker in 1:n_workers
]

@testset "non-backtracking spectrum" begin
    forest = nb_test_matrix(3, 3, [(1, 1), (2, 1), (2, 2), (3, 2), (3, 3)])
    forest_spectrum = nb_spectrum(forest; seed=7)
    @test forest_spectrum.converged
    @test forest_spectrum.lambda_nb == 0.0
    @test forest_spectrum.circuit_rank == 0
    @test isempty(forest_spectrum.components)
    @test all(iszero, forest_spectrum.node_scores)
    @test all(==(-1), forest_spectrum.distance_to_core)

    cycle = nb_test_matrix(2, 2, complete_bipartite_edges(2, 2))
    cycle_spectrum = nb_spectrum(cycle; seed=7)
    @test cycle_spectrum.lambda_nb == 1.0
    @test cycle_spectrum.circuit_rank == 1
    @test all(cycle_spectrum.in_two_core)
    @test all(iszero, cycle_spectrum.distance_to_core)
    @test length(cycle_spectrum.components) == 1
    @test sum(cycle_spectrum.node_scores) ≈ 1.0

    for size in (3, 4)
        complete = nb_test_matrix(size, size, complete_bipartite_edges(size, size))
        spectrum = nb_spectrum(complete; seed=11)
        @test spectrum.converged
        @test spectrum.lambda_nb ≈ size - 1 atol=1e-10
        @test spectrum.circuit_rank == (size - 1)^2
        @test sum(spectrum.node_scores) ≈ 1.0
        @test all(>=(0), spectrum.node_scores)
    end

    disconnected_edges = vcat(
        complete_bipartite_edges(3, 3),
        complete_bipartite_edges(2, 2; firm_offset=3, worker_offset=3),
    )
    disconnected = nb_test_matrix(5, 5, disconnected_edges)
    disconnected_spectrum = nb_spectrum(disconnected; seed=19)
    @test disconnected_spectrum.converged
    @test disconnected_spectrum.lambda_nb ≈ 2.0 atol=1e-10
    @test length(disconnected_spectrum.components) == 2
    @test sort(getfield.(disconnected_spectrum.components, :lambda_nb)) ≈ [1.0, 2.0]
    @test disconnected_spectrum.binding_component == 1
    for component in disconnected_spectrum.components
        @test sum(disconnected_spectrum.node_scores[component.nodes]) ≈ 1.0
    end

    # Force the iterative path on a small graph to verify that the explicit
    # Arnoldi start vector makes the complete result reproducible.
    complete = nb_test_matrix(4, 4, complete_bipartite_edges(4, 4))
    first_run = nb_spectrum(complete; seed=2026, dense_threshold=0)
    second_run = nb_spectrum(complete; seed=2026, dense_threshold=0)
    @test first_run.converged
    @test first_run.lambda_nb ≈ 3.0 atol=1e-6
    @test first_run.lambda_nb == second_run.lambda_nb
    @test first_run.eigenvalues ≈ second_run.eigenvalues atol=1e-14
    @test first_run.node_scores ≈ second_run.node_scores atol=1e-14

    df = DataFrame(
        firm_id=first.(complete_bipartite_edges(3, 3)),
        worker_id=last.(complete_bipartite_edges(3, 3)),
        y=collect(1.0:9.0),
    )
    problem = GMRFProblem(df; standardize=false)
    @test nb_spectrum(problem; seed=3).lambda_nb ≈ 2.0 atol=1e-10

    @test_throws ArgumentError nb_spectrum(cycle; nev=0)
    @test_throws ArgumentError nb_spectrum(cycle; tol=0.0)

    failed = BipartiteGMRF.nb_dense_component(sparse([1], [1], [NaN], 2, 2), 1, 1)
    @test !failed.converged
    @test isnan(failed.lambda_nb)
    @test all(isnan, failed.node_scores)
end
