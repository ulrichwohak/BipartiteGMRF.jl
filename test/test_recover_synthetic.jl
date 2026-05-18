using Random

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
    truth = (rho=0.45, sigma_a=0.8, sigma_z=0.5, sigma_epsilon=0.25)
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

@testset "synthetic recovery" begin
    seeds = [203]
    estimates = map(seeds) do seed
        df, _ = simulate_gmrf_panel(seed)
        gmrf_mle(
            df;
            standardize=false,
            solver=ExactCholesky(optim_iters=160, polish=true),
            decompose=false,
            seed=seed,
        )
    end
    _, truth = simulate_gmrf_panel(first(seeds))

    rhos = [estimate.rho for estimate in estimates]
    sigma_as = [estimate.sigma_a for estimate in estimates]
    sigma_zs = [estimate.sigma_z for estimate in estimates]
    sigma_epsilons = [estimate.sigma_epsilon for estimate in estimates]

    @test all(rho -> abs(rho - truth.rho) < 0.10, rhos)
    @test all(s -> abs(log(s) - log(truth.sigma_a)) < 0.15, sigma_as)
    @test all(s -> abs(log(s) - log(truth.sigma_z)) < 0.15, sigma_zs)
    @test all(s -> abs(log(s) - log(truth.sigma_epsilon)) < 0.10, sigma_epsilons)
end
