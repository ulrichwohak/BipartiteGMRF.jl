# Criterion-1 follow-up (issue #112, plan §0): does an INTERIOR consistent root
# exist when EM starts at the truth?
#
# The likelihood is globally unbounded (degenerate corner, probe_unbounded.jl),
# but the model IS identified in population when ρ ≠ 0: Ω is block-diagonal by
# firm, so cross-firm covariances of Σ = AK⁻¹A' + Ω pin down σ_a. Mixture-style
# resolution: an interior local root centered at truth may exist; the collapse
# in dev/mc_identification.jl may just be the default init landing in the
# degenerate basin. Discriminator: same DGP, EM initialized at the truth
# (θ true, Ω_i true). If estimates stay interior and centered → local-root
# story confirmed; if they slide to σ̂ = 0 anyway → the basin of the degenerate
# corner swallows the truth and a penalty is unavoidable.
using BipartiteGMRF, Random, LinearAlgebra, SparseArrays, Statistics, Printf

const N = 60                  # firms = workers (caterpillar: f1-w1-f2-w2-…)
const N_SEEDS = 25
const TRUTH = (rho = 0.4, sigma_a = 0.7, sigma_z = 0.5,
               sigma_epsilon = 0.2, rho_corr = 0.5)
const EIG_FLOOR = 1e-3
const MAX_ITER = 200
const FTOL = 1e-6

# Caterpillar forest (acyclic): f_i—w_i, w_i—f_{i+1}. Same DGP as
# dev/mc_identification.jl, only the EM init differs.
f = Int[]; w = Int[]
for i in 1:N
    push!(f, i); push!(w, i)
    i < N && (push!(f, i + 1); push!(w, i))
end
A = sparse(f, w, ones(length(f)), N, N)
model_true = BipartiteVarianceStableModel(A; rho_limit = 0.99)
Q = BipartiteGMRF.model_precision(model_true, TRUTH.rho, TRUTH.sigma_a, TRUTH.sigma_z)
L = cholesky(Symmetric(Matrix(Q))).L

rows_of = [findall(==(i), f) for i in 1:N]
block_cov = [begin
    m = length(rows_of[i])
    TRUTH.sigma_epsilon^2 .* (Matrix{Float64}(LinearAlgebra.I, m, m) .+
        TRUTH.rho_corr .* (ones(m, m) .- Matrix{Float64}(LinearAlgebra.I, m, m)))
end for i in 1:N]
block_chol = [cholesky(Symmetric(C)).L for C in block_cov]

r_hat = Float64[]; sa_hat = Float64[]; sz_hat = Float64[]; se_hat = Float64[]
nconv = 0
ninterior = 0
for seed in 1:N_SEEDS
    rng = MersenneTwister(seed)
    theta = transpose(L) \ randn(rng, 2N)
    eps = zeros(length(f))
    for i in 1:N
        rr = rows_of[i]
        eps[rr] = block_chol[i] * randn(rng, length(rr))
    end
    y = [theta[f[k]] + theta[N + w[k]] + eps[k] for k in eachindex(f)]

    ss = suffstats(BipartiteVarianceStableModel, f, w, y;
        weighting = Weighting(observations = :raw), standardize = true,
        error_blocks = :free, firm_group = f)
    model = BipartiteVarianceStableModel(ss.A_prior; rho_limit = 0.99)
    s = ss.y_std
    fit = BipartiteGMRF.optimize_emfree(model, ss,
        EMFreeBlocks(max_iter = MAX_ITER, ftol = FTOL, eig_floor = EIG_FLOOR);
        init_theta = (TRUTH.rho, TRUTH.sigma_a / s, TRUTH.sigma_z / s),
        init_blocks = [C ./ s^2 for C in block_cov])
    push!(r_hat, fit.rho); push!(sa_hat, fit.sigma_a)
    push!(sz_hat, fit.sigma_z); push!(se_hat, fit.sigma_epsilon)
    fit.converged && (global nconv += 1)
    fit.sigma_a > 1e-3 && fit.sigma_z > 1e-3 && (global ninterior += 1)
end

function summarize(tag, est, truth)
    println(rpad(tag, 14), " truth=", round(truth; digits = 3),
            "  mean=", round(mean(est); digits = 3),
            "  bias=", round(mean(est) - truth; digits = 3),
            "  sd=", round(std(est); digits = 3),
            "  min=", round(minimum(est); digits = 3),
            "  max=", round(maximum(est); digits = 3))
end
println("=== TRUTH-INIT EM, caterpillar N=$N, $(length(f)) edges, $N_SEEDS draws ===")
println("converged: $nconv / $N_SEEDS   interior (σ̂a, σ̂z > 1e-3): $ninterior / $N_SEEDS")
summarize("rho", r_hat, TRUTH.rho)
summarize("sigma_a", sa_hat, TRUTH.sigma_a)
summarize("sigma_z", sz_hat, TRUTH.sigma_z)
summarize("sigma_eps", se_hat, TRUTH.sigma_epsilon)  # bar σ² = tr(Ω)/K, unaffected by within-block correlation
