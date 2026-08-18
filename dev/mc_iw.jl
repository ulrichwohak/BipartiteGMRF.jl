# Criterion-4 MC for the INTEGRATED free-block solver (EMIWBlocks, remedy E).
# Same DGPs that made the profile-MLE collapse (dev/mc_identification.jl,
# dev/mc_dense_graph.jl): fixed equicorrelated within-firm error blocks
# (r = 0.5), redraw a, z, ε. The integrated model treats Ω_i as IW draws; the
# fixed-Ω DGP is its δ → ∞ limit, so expect δ̂ large, r̂ ≈ 0.5, and — the point
# of the exercise — (ρ̂, σ̂_a, σ̂_z) centered at truth where the profile MLE
# collapsed to zero.
using BipartiteGMRF, Random, LinearAlgebra, SparseArrays, Statistics, Printf

const N_SEEDS = 25
const SOLVER = EMIWBlocks(max_iter = 800, ftol = 1e-9)

function build_caterpillar(N)
    f = Int[]; w = Int[]
    for i in 1:N
        push!(f, i); push!(w, i)
        i < N && (push!(f, i + 1); push!(w, i))
    end
    return f, w, N, N
end

function build_dense(NF, M_I, NW_pool)
    rng_g = MersenneTwister(2026)
    f = Int[]; w = Int[]
    for i in 1:NF
        for j in shuffle(rng_g, 1:NW_pool)[1:M_I]
            push!(f, i); push!(w, j)
        end
    end
    wl = sort(unique(w)); wmap = Dict(v => k for (k, v) in enumerate(wl))
    w = [wmap[v] for v in w]
    return f, w, NF, length(wl)
end

function run_mc(tag, f, w, NF, NW; rho_limit_spec, rho_frac = 0.75)
    A = sparse(f, w, ones(length(f)), NF, NW)
    model0 = BipartiteVarianceStableModel(A; rho_limit = rho_limit_spec)
    lim = BipartiteGMRF.rho_limit(model0)
    truth = (rho = rho_frac * lim, sigma_a = 0.7, sigma_z = 0.5,
             sigma_epsilon = 0.2, r = 0.5)
    Q = BipartiteGMRF.model_precision(model0, truth.rho, truth.sigma_a, truth.sigma_z)
    L = cholesky(Symmetric(Matrix(Q))).L
    rows_of = [findall(==(i), f) for i in 1:NF]
    block_chol = [begin
        m = length(rows_of[i])
        C = truth.sigma_epsilon^2 .* ((1 - truth.r) .* Matrix{Float64}(I, m, m) .+ truth.r)
        cholesky(Symmetric(C)).L
    end for i in 1:NF]

    est = (rho = Float64[], sa = Float64[], sz = Float64[], se = Float64[],
           r = Float64[], delta = Float64[])
    nconv = 0
    for seed in 1:N_SEEDS
        rng = MersenneTwister(seed)
        theta = transpose(L) \ randn(rng, NF + NW)
        eps = zeros(length(f))
        for i in 1:NF
            rr = rows_of[i]
            eps[rr] = block_chol[i] * randn(rng, length(rr))
        end
        y = [theta[f[k]] + theta[NF + w[k]] + eps[k] for k in eachindex(f)]
        res = fit_mle(BipartiteVarianceStableModel, f, w, y;
            weighting = Weighting(observations = :raw), standardize = true,
            error_blocks = :free, firm_group = f, solver = SOLVER,
            rho_limit = rho_limit_spec)
        push!(est.rho, res.rho); push!(est.sa, res.sigma_a)
        push!(est.sz, res.sigma_z); push!(est.se, res.sigma_epsilon)
        push!(est.r, res.metadata.error_corr_r); push!(est.delta, res.metadata.t_dof_delta)
        res.converged && (nconv += 1)
    end
    println("\n=== $tag: $(NF)×$(NW), $(length(f)) edges, rho_true=$(round(truth.rho; digits=3)), $nconv/$N_SEEDS converged ===")
    for (name, v, t) in (("rho", est.rho, truth.rho), ("sigma_a", est.sa, truth.sigma_a),
                         ("sigma_z", est.sz, truth.sigma_z), ("sigma_eps", est.se, truth.sigma_epsilon),
                         ("r", est.r, truth.r))
        println(rpad(name, 10), " truth=", rpad(round(t; digits = 3), 6),
                " mean=", rpad(round(mean(v); digits = 3), 7),
                " bias=", rpad(round(mean(v) - t; digits = 3), 7),
                " sd=", round(std(v); digits = 3))
    end
    println(rpad("delta", 10), " (∞ in DGP)  median=", round(median(est.delta); digits = 1))
end

f, w, NF, NW = build_caterpillar(60)
run_mc("caterpillar tree", f, w, NF, NW; rho_limit_spec = 0.99, rho_frac = 0.4 / 0.99)
f, w, NF, NW = build_dense(40, 5, 100)
run_mc("dense graph", f, w, NF, NW; rho_limit_spec = :auto)
