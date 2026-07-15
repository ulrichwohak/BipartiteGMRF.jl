using Random: MersenneTwister, rand

function theory_adjacency(A_prior)
    n_firms, n_workers = size(A_prior)
    W = Float64.(Matrix(A_prior .!= 0))
    return [zeros(n_firms, n_firms) W; transpose(W) zeros(n_workers, n_workers)]
end

function theory_kernel(adjacency, rho)
    n = size(adjacency, 1)
    identity = Matrix{Float64}(I, n, n)
    degree = Diagonal(vec(sum(adjacency; dims=2)))
    return identity - rho .* adjacency + rho^2 .* (degree - identity)
end

function truncated_nb_walk_sum(adjacency, rho; atol=1e-14, max_length=500)
    n = size(adjacency, 1)
    identity = Matrix{Float64}(I, n, n)
    degree = Matrix(Diagonal(vec(sum(adjacency; dims=2))))
    previous_one = copy(adjacency)
    total = identity + rho .* previous_one

    current = previous_one * adjacency - degree
    rho_power = rho^2
    term = rho_power .* current
    total .+= term
    maximum(abs, term) <= atol && return total, 2, true

    degree_minus_identity = degree - identity
    for walk_length in 3:max_length
        next = current * adjacency - previous_one * degree_minus_identity
        rho_power *= rho
        term = rho_power .* next
        total .+= term
        maximum(abs, term) <= atol && return total, walk_length, true
        previous_one, current = current, next
    end
    return total, max_length, false
end

function seeded_theory_graph(seed)
    rng = MersenneTwister(seed)
    edges = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for firm in 1:4, worker in 1:4
        rand(rng) < 0.3 && push!(edges, (firm, worker))
    end
    unique!(edges)
    return nb_test_matrix(4, 4, edges)
end

@testset "non-backtracking walk theory" begin
    for seed in (93, 930)
        A_prior = seeded_theory_graph(seed)
        adjacency = theory_adjacency(A_prior)
        spectrum = nb_spectrum(A_prior; seed=seed)
        rho = 0.4 / max(1.0, spectrum.lambda_nb)
        partial, truncation_length, converged = truncated_nb_walk_sum(adjacency, rho)
        reference = (1.0 - rho^2) .* inv(theory_kernel(adjacency, rho))
        @test converged
        @test truncation_length < 500
        @test partial ≈ reference atol=5e-13 rtol=5e-13
    end

    path = nb_test_matrix(2, 2, [(1, 1), (2, 1), (2, 2)])
    path_adjacency = theory_adjacency(path)
    rho = 0.4
    path_generating = (1.0 - rho^2) .* inv(theory_kernel(path_adjacency, rho))
    path_distances = [
        0 2 1 3
        2 0 1 1
        1 1 0 2
        3 1 2 0
    ]
    @test path_generating ≈ rho .^ path_distances atol=1e-13

    cycle = nb_test_matrix(2, 2, complete_bipartite_edges(2, 2))
    cycle_adjacency = theory_adjacency(cycle)
    cycle_inverse = inv(theory_kernel(cycle_adjacency, rho))
    expected_inflation = (1.0 + rho^4) / (1.0 - rho^4)
    expected_correlation = (rho + rho^3) / (1.0 + rho^4)
    @test all(diag(cycle_inverse) .* (1.0 - rho^2) .≈ expected_inflation)
    @test cycle_inverse[1, 3] /
        sqrt(cycle_inverse[1, 1] * cycle_inverse[3, 3]) ≈ expected_correlation

    complete = nb_test_matrix(3, 3, complete_bipartite_edges(3, 3))
    for (A_prior, boundary) in ((cycle, 1.0), (complete, 0.5))
        kernel = Symmetric(theory_kernel(theory_adjacency(A_prior), boundary))
        @test minimum(abs, eigvals(kernel)) < 1e-12
        @test rank(Matrix(kernel); atol=1e-10) < size(kernel, 1)
    end
end
