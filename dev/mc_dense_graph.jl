# Criterion-1/4 follow-up (issue #112, plan §0): does a DENSE graph create the
# interior consistent root that the caterpillar lacks?
#
# On a tree, cross-firm information is minimal (N−1 firm–firm links however
# large the blocks), and truth-init EM still slides toward the degenerate
# corner (dev/mc_truth_init.jl: σ̂a → 0). Identification of σ_a runs through
# cross-firm covariances (Ω is block-diagonal by firm, so it cannot produce
# them) — a graph with many worker-shared firm pairs carries far more of them.
# Here: random bipartite graph, firms with 5 edges each, most workers shared
# by 2 firms; ρ ceiling from the NB spectrum (rho_limit = :auto). EM run from
# (i) default init and (ii) truth init on the same draws.
using BipartiteGMRF, Random, LinearAlgebra, SparseArrays, Statistics, Printf

const NF = 40                 # firms, 5 edges each → 200 edges
const M_I = 5
const N_SEEDS = 25
const EIG_FLOOR = 1e-3
const MAX_ITER = 200
const FTOL = 1e-6

# Random bipartite graph: 100 workers; each firm picks 5 distinct workers.
# Worker pool sized so most workers serve 2 firms → dense cross-firm linkage.
rng_g = MersenneTwister(2026)
NW = 100
f = Int[]; w = Int[]
for i in 1:NF
    for j in shuffle(rng_g, 1:NW)[1:M_I]
        push!(f, i); push!(w, j)
    end
end
# keep only workers that appear (relabel densely)
wl = sort(unique(w)); wmap = Dict(v => k for (k, v) in enumerate(wl))
w = [wmap[v] for v in w]
NWeff = length(wl)
A = sparse(f, w, ones(length(f)), NF, NWeff)

model_true = BipartiteVarianceStableModel(A; rho_limit = :auto)
lim = BipartiteGMRF.rho_limit(model_true)
TRUTH = (rho = 0.75 * lim, sigma_a = 0.7, sigma_z = 0.5,
         sigma_epsilon = 0.2, rho_corr = 0.5)
@printf("graph: %d firms × %d workers, %d edges; rho ceiling (auto) = %.4f, rho_true = %.4f\n",
        NF, NWeff, length(f), lim, TRUTH.rho)

Q = BipartiteGMRF.model_precision(model_true, TRUTH.rho, TRUTH.sigma_a, TRUTH.sigma_z)
L = cholesky(Symmetric(Matrix(Q))).L

rows_of = [findall(==(i), f) for i in 1:NF]
block_cov = [begin
    m = length(rows_of[i])
    TRUTH.sigma_epsilon^2 .* (Matrix{Float64}(LinearAlgebra.I, m, m) .+
        TRUTH.rho_corr .* (ones(m, m) .- Matrix{Float64}(LinearAlgebra.I, m, m)))
end for i in 1:NF]
block_chol = [cholesky(Symmetric(C)).L for C in block_cov]

function run_mc(; truth_init::Bool)
    r_hat = Float64[]; sa_hat = Float64[]; sz_hat = Float64[]; se_hat = Float64[]
    nconv = 0
    for seed in 1:N_SEEDS
        rng = MersenneTwister(seed)
        theta = transpose(L) \ randn(rng, NF + NWeff)
        eps = zeros(length(f))
        for i in 1:NF
            rr = rows_of[i]
            eps[rr] = block_chol[i] * randn(rng, length(rr))
        end
        y = [theta[f[k]] + theta[NF + w[k]] + eps[k] for k in eachindex(f)]

        ss = suffstats(BipartiteVarianceStableModel, f, w, y;
            weighting = Weighting(observations = :raw), standardize = true,
            error_blocks = :free, firm_group = f)
        model = BipartiteVarianceStableModel(ss.A_prior; rho_limit = lim)
        s = ss.y_std
        fit = BipartiteGMRF.optimize_emfree(model, ss,
            EMFreeBlocks(max_iter = MAX_ITER, ftol = FTOL, eig_floor = EIG_FLOOR);
            init_theta = truth_init ? (TRUTH.rho, TRUTH.sigma_a / s, TRUTH.sigma_z / s) : nothing,
            init_blocks = truth_init ? [C ./ s^2 for C in block_cov] : nothing)
        push!(r_hat, fit.rho); push!(sa_hat, fit.sigma_a)
        push!(sz_hat, fit.sigma_z); push!(se_hat, fit.sigma_epsilon)
        fit.converged && (nconv += 1)
    end
    return r_hat, sa_hat, sz_hat, se_hat, nconv
end

function summarize(tag, est, truth)
    println(rpad(tag, 14), " truth=", round(truth; digits = 3),
            "  mean=", round(mean(est); digits = 3),
            "  bias=", round(mean(est) - truth; digits = 3),
            "  sd=", round(std(est); digits = 3),
            "  min=", round(minimum(est); digits = 3),
            "  max=", round(maximum(est); digits = 3))
end

for truth_init in (true, false)
    r, sa, sz, se, nconv = run_mc(; truth_init)
    println("\n=== dense graph, ", truth_init ? "TRUTH init" : "DEFAULT init",
            " ($nconv / $N_SEEDS converged) ===")
    summarize("rho", r, TRUTH.rho)
    summarize("sigma_a", sa, TRUTH.sigma_a)
    summarize("sigma_z", sz, TRUTH.sigma_z)
    summarize("sigma_eps", se, TRUTH.sigma_epsilon)
end
