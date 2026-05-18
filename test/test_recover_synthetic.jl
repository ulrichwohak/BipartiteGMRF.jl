using Random
using Statistics

function connected_random_edges(rng::AbstractRNG, n_firms::Int, n_workers::Int, n_edges::Int)
    edges = Set{Tuple{Int,Int}}()
    for i in 1:n_firms
        push!(edges, (i, rand(rng, 1:n_workers)))
    end
    for j in 1:n_workers
        push!(edges, (rand(rng, 1:n_firms), j))
    end
    while length(edges) < n_edges
        push!(edges, (rand(rng, 1:n_firms), rand(rng, 1:n_workers)))
    end
    return sort!(collect(edges))
end

function simulate_gmrf_panel(seed::Int; n_firms::Int=30, n_workers::Int=30, n_edges::Int=180, reps::Int=3)
    rng = MersenneTwister(seed)
    edges = connected_random_edges(rng, n_firms, n_workers, n_edges)
    firm_id = Int[]
    worker_id = Int[]
    for (firm, worker) in edges, _ in 1:reps
        push!(firm_id, firm)
        push!(worker_id, 100 + worker)
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
    seeds = [202, 203, 204, 205]
    estimates = map(seeds) do seed
        df, _ = simulate_gmrf_panel(seed)
        gmrf_mle(
            df;
            standardize=false,
            solver=ExactCholesky(optim_iters=120, polish=true),
            decompose=false,
            seed=seed,
        )
    end
    _, truth = simulate_gmrf_panel(first(seeds))

    rhos = [estimate.rho for estimate in estimates]
    sigma_as = [estimate.sigma_a for estimate in estimates]
    sigma_zs = [estimate.sigma_z for estimate in estimates]
    sigma_epsilons = [estimate.sigma_epsilon for estimate in estimates]

    @test count(>(0), rhos) >= 3
    @test abs(mean(rhos) - truth.rho) < 0.30
    @test abs(mean(log.(sigma_as)) - log(truth.sigma_a)) < 0.30
    @test abs(mean(log.(sigma_zs)) - log(truth.sigma_z)) < 0.40
    @test abs(mean(log.(sigma_epsilons)) - log(truth.sigma_epsilon)) < 0.10
end
