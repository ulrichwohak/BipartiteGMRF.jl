using Random

# Parallel (firm index, worker index, outcome) observation vectors — the
# package's data interface.

function synthetic_data()
    return (
        f = [1, 1, 2, 2, 3, 3, 1, 2],
        w = [1, 2, 2, 3, 3, 4, 4, 1],
        y = [1.2, 0.7, 0.9, 1.5, 1.1, 0.4, 1.0, 0.8],
    )
end

function repeated_data()
    return (
        f = [1, 1, 1, 2, 2, 3, 3, 1, 2, 2],
        w = [1, 1, 2, 2, 3, 3, 4, 4, 1, 1],
        y = [1.2, 1.1, 0.7, 0.9, 1.5, 1.1, 0.4, 1.0, 0.8, 0.85],
    )
end

function suffstats_synthetic(M=BipartiteNormalizedModel; kwargs...)
    d = synthetic_data()
    return suffstats(M, d.f, d.w, d.y; kwargs...)
end

function suffstats_repeated(M=BipartiteNormalizedModel; kwargs...)
    d = repeated_data()
    return suffstats(M, d.f, d.w, d.y; kwargs...)
end

function fitted_exact()
    d = synthetic_data()
    return fit_mle(BipartiteNormalizedModel, d.f, d.w, d.y;
        solver=ExactCholesky(optim_iters=5, polish=false),
        seed=1,
        verbose=false,
    )
end

# Acyclic firm-worker graph (a caterpillar path: f1-w1-f2-w2-...-f_n-w_n), used
# to exercise the BipartiteVarianceStableModel + ExactCholesky path, which is
# only valid on forests.
function tree_data(; n=12, seed=11)
    rng = MersenneTwister(seed)
    firm = Int[]
    worker = Int[]
    for i in 1:n
        push!(firm, i); push!(worker, i)                 # f_i — w_i
        i < n && (push!(firm, i + 1); push!(worker, i))  # w_i — f_{i+1}
    end
    return (f = firm, w = worker, y = randn(rng, length(firm)))
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
    f_idx = Int[]
    w_idx = Int[]
    sizehint!(f_idx, length(edges) * reps)
    sizehint!(w_idx, length(edges) * reps)
    for (firm, worker) in edges, _ in 1:reps
        push!(f_idx, firm)
        push!(w_idx, worker)
    end

    ss = suffstats(BipartiteNormalizedModel, f_idx, w_idx, zeros(length(f_idx));
        n_firms=n_firms, n_workers=n_workers, standardize=false)
    model = BipartiteNormalizedModel(ss.A_prior; rho_limit=0.99)
    Q = BipartiteGMRF.model_precision(model, truth.rho, truth.sigma_a, truth.sigma_z)
    L = cholesky(Symmetric(Matrix(Q))).L
    theta = transpose(L) \ randn(rng, n_firms + n_workers)
    y = [
        theta[f_idx[k]] + theta[n_firms + w_idx[k]] + truth.sigma_epsilon * randn(rng)
        for k in eachindex(f_idx)
    ]
    return (f = f_idx, w = w_idx, y = y), truth
end
