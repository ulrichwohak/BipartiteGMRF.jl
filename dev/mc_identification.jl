# Criterion-4 Monte-Carlo gate (issue #112), tree version.
# Fixed caterpillar forest (the variance-stable precision's domain), redraw
# a, z, ε many times; ε has within-firm equicorrelation (a non-degenerate
# free-Ω DGP). Check (ρ̂, σ̂a, σ̂z) centering, stratified by convergence.
using BipartiteGMRF, Random, LinearAlgebra, SparseArrays, Statistics, Printf

const N = 60                  # firms = workers (caterpillar: f1-w1-f2-w2-…)
const N_SEEDS = 25
const TRUTH = (rho = 0.4, sigma_a = 0.7, sigma_z = 0.5,
               sigma_epsilon = 0.2, rho_corr = 0.5)
const EIG_FLOOR = 1e-3
const MAX_ITER = 200
const FTOL = 1e-6

# Caterpillar forest (acyclic): f_i—w_i, w_i—f_{i+1}.
f = Int[]; w = Int[]
for i in 1:N
    push!(f, i); push!(w, i)
    i < N && (push!(f, i + 1); push!(w, i))
end
A = sparse(f, w, ones(length(f)), N, N)
model = BipartiteVarianceStableModel(A; rho_limit = 0.99)
Q = BipartiteGMRF.model_precision(model, TRUTH.rho, TRUTH.sigma_a, TRUTH.sigma_z)
L = cholesky(Symmetric(Matrix(Q))).L

# Per-firm error blocks: equicorrelated ρ_corr (≡ AR(1) for m_i=2).
rows_of = [findall(==(i), f) for i in 1:N]
block_chol = [begin
    m = length(rows_of[i])
    if m == 1
        reshape([TRUTH.sigma_epsilon], 1, 1)
    else
        R = TRUTH.sigma_epsilon^2 .* (Matrix{Float64}(LinearAlgebra.I, m, m) .+
             TRUTH.rho_corr .* (ones(m, m) .- Matrix{Float64}(LinearAlgebra.I, m, m)))
        cholesky(Symmetric(R)).L
    end
end for i in 1:N]

allr = Float64[]; alls_a = Float64[]; alls_z = Float64[]; alls_e = Float64[]
cr = Float64[]; cs_a = Float64[]; cs_z = Float64[]; cs_e = Float64[]
nconv = 0
for seed in 1:N_SEEDS
    rng = MersenneTwister(seed)
    theta = transpose(L) \ randn(rng, 2N)
    eps = zeros(length(f))
    for i in 1:N
        rr = rows_of[i]
        eps[rr] = block_chol[i] * randn(rng, length(rr))
    end
    y = [theta[f[k]] + theta[N + w[k]] + eps[k] for k in eachindex(f)]
    res = fit_mle(BipartiteVarianceStableModel, f, w, y;
        weighting = Weighting(observations = :raw), standardize = true,
        error_blocks = :free, firm_group = f,
        solver = EMFreeBlocks(max_iter = MAX_ITER, ftol = FTOL, eig_floor = EIG_FLOOR))
    push!(allr, res.rho); push!(alls_a, res.sigma_a)
    push!(alls_z, res.sigma_z); push!(alls_e, res.sigma_epsilon)
    if res.converged
        global nconv += 1
        push!(cr, res.rho); push!(cs_a, res.sigma_a)
        push!(cs_z, res.sigma_z); push!(cs_e, res.sigma_epsilon)
    end
end

function summarize(tag, est, truth)
    println(rpad(tag, 14), " truth=", round(truth; digits = 3),
            "  mean=", round(mean(est); digits = 3),
            "  bias=", round(mean(est) - truth; digits = 3),
            "  sd=", round(std(est); digits = 3))
end
println("=== caterpillar tree: N=$N, $(length(f)) edges, $N_SEEDS draws, eig_floor=$EIG_FLOOR ===")
println("ALL seeds:")
summarize("rho", allr, TRUTH.rho)
summarize("sigma_a", alls_a, TRUTH.sigma_a)
summarize("sigma_z", alls_z, TRUTH.sigma_z)
summarize("sigma_eps", alls_e, TRUTH.sigma_epsilon)
println("CONVERGED only ($nconv / $N_SEEDS):")
summarize("rho", cr, TRUTH.rho)
summarize("sigma_a", cs_a, TRUTH.sigma_a)
summarize("sigma_z", cs_z, TRUTH.sigma_z)
summarize("sigma_eps", cs_e, TRUTH.sigma_epsilon)