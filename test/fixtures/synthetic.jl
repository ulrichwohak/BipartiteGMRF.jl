using Random

function synthetic_df()
    return DataFrame(
        firm_id = [1, 1, 2, 2, 3, 3, 1, 2],
        worker_id = [10, 11, 11, 12, 12, 13, 13, 10],
        y = [1.2, 0.7, 0.9, 1.5, 1.1, 0.4, 1.0, 0.8],
    )
end

function repeated_df()
    return DataFrame(
        firm_id = [1, 1, 1, 2, 2, 3, 3, 1, 2, 2],
        worker_id = [10, 10, 11, 11, 12, 12, 13, 13, 10, 10],
        y = [1.2, 1.1, 0.7, 0.9, 1.5, 1.1, 0.4, 1.0, 0.8, 0.85],
    )
end

function custom_column_df()
    return DataFrame(
        employer = ["a", "a", "b", "b", "c", "c"],
        person = ["p1", "p2", "p2", "p3", "p3", "p4"],
        outcome = [1.0, missing, 0.8, 1.1, 0.9, 1.2],
    )
end

function fitted_exact(; decompose=false)
    return gmrf_mle(
        synthetic_df();
        solver=ExactCholesky(optim_iters=5, polish=false),
        decompose=decompose,
        seed=1,
        verbose=false,
    )
end

function connected_random_edges(rng::AbstractRNG, n_firms::Int, n_workers::Int, n_edges::Int)
    edges = Set{Tuple{Int,Int}}()
    for firm in 1:n_firms
        push!(edges, (firm, rand(rng, 1:n_workers)))
    end
    for worker in 1:n_workers
        push!(edges, (rand(rng, 1:n_firms), worker))
    end
    while length(edges) < n_edges
        push!(edges, (rand(rng, 1:n_firms), rand(rng, 1:n_workers)))
    end
    return sort!(collect(edges))
end

function simulate_gmrf_panel(
    seed::Int;
    n_firms::Int=300,
    n_workers::Int=300,
    n_edges::Int=6000,
    reps::Int=5,
    truth=(rho=0.45, sigma_a=0.8, sigma_z=0.5, sigma_epsilon=0.25),
)
    rng = MersenneTwister(seed)
    edges = connected_random_edges(rng, n_firms, n_workers, n_edges)
    firm_id = Int[]
    worker_id = Int[]
    sizehint!(firm_id, length(edges) * reps)
    sizehint!(worker_id, length(edges) * reps)
    for (firm, worker) in edges, _ in 1:reps
        push!(firm_id, firm)
        push!(worker_id, 1000 + worker)
    end

    template = DataFrame(firm_id=firm_id, worker_id=worker_id, y=zeros(length(firm_id)))
    problem = GMRFProblem(template; standardize=false)
    Q = BipartiteGMRF.precision_matrix(problem, truth.rho, truth.sigma_a, truth.sigma_z)
    L = cholesky(Symmetric(Matrix(Q))).L
    theta = transpose(L) \ randn(rng, n_firms + n_workers)
    y = [
        theta[problem.firm_to_index[firm_id[k]]] +
        theta[problem.N_firms + problem.worker_to_index[worker_id[k]]] +
        truth.sigma_epsilon * randn(rng)
        for k in eachindex(firm_id)
    ]
    return DataFrame(firm_id=firm_id, worker_id=worker_id, y=y), truth
end
