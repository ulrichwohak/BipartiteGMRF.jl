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

    cycle_with_leaf = nb_test_matrix(
        3,
        2,
        vcat(complete_bipartite_edges(2, 2), [(3, 1)]),
    )
    @test nb_spectrum(cycle_with_leaf).distance_to_core == [0, 0, 1, 0, 0]

    # Arnoldi may converge one extra Ritz value to preserve a real Schur block.
    # The public result is capped at nev; compare the identified spectrum rather
    # than ordering inside a repeated-modulus eigenspace.
    complete = nb_test_matrix(4, 4, complete_bipartite_edges(4, 4))
    first_run = nb_spectrum(complete; seed=2026, dense_threshold=0)
    second_run = nb_spectrum(complete; seed=2026, dense_threshold=0)
    @test first_run.converged
    @test first_run.lambda_nb ≈ 3.0 atol=1e-6
    @test first_run.lambda_nb == second_run.lambda_nb
    @test length(first_run.eigenvalues) == length(second_run.eigenvalues) == 6
    @test sort(abs.(first_run.eigenvalues)) ≈
          sort(abs.(second_run.eigenvalues)) atol=1e-14
    @test first_run.node_scores ≈ second_run.node_scores atol=1e-14

    irregular = nb_test_matrix(
        4,
        3,
        vcat(complete_bipartite_edges(3, 3), [(4, 1), (4, 2)]),
    )
    irregular_first = nb_spectrum(irregular; seed=44, nev=1, dense_threshold=0)
    irregular_second = nb_spectrum(irregular; seed=44, nev=1, dense_threshold=0)
    @test irregular_first.lambda_nb == irregular_second.lambda_nb
    @test irregular_first.node_scores ≈ irregular_second.node_scores atol=1e-14
    @test maximum(irregular_first.node_scores) - minimum(irregular_first.node_scores) > 0.01

    A_complete33 = sparse(
        first.(complete_bipartite_edges(3, 3)),
        last.(complete_bipartite_edges(3, 3)),
        ones(Float64, 9),
        3, 3,
    )
    @test nb_spectrum(A_complete33; seed=3).lambda_nb ≈ 2.0 atol=1e-10

    @test_throws ArgumentError nb_spectrum(cycle; nev=0)
    @test_throws ArgumentError nb_spectrum(cycle; tol=0.0)
    @test_throws ArgumentError nb_spectrum(cycle; dense_threshold=-1)
    @test_throws ArgumentError nb_spectrum(cycle; arnoldi_restarts=-1)

    failed = BipartiteGMRF.nb_dense_component(sparse([1], [1], [NaN], 2, 2), 1, 1)
    @test !failed.converged
    @test isnan(failed.lambda_nb)
    @test all(isnan, failed.node_scores)

    failed_arnoldi = BipartiteGMRF.nb_arnoldi_component(
        spzeros(1, 2),
        1,
        1,
        7,
        1e-7,
        1,
    )
    @test !failed_arnoldi.converged
    @test isnan(failed_arnoldi.lambda_nb)
    @test all(isnan, failed_arnoldi.node_scores)

    not_converged = nb_spectrum(
        complete;
        nev=6,
        dense_threshold=0,
        arnoldi_restarts=0,
    )
    @test !not_converged.converged
    @test isnan(not_converged.lambda_nb)
    @test not_converged.binding_component == 0
    @test all(isnan, not_converged.node_scores)
end
