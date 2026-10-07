# Examples

Throughout, `f`, `w`, and `y` are parallel vectors: `f[k]` and `w[k]` are the
1-based firm and worker indices of observation `k`, and `y[k]` its outcome.
If your data lives in a table with entity identifiers, map the identifiers to
dense indices first, e.g.

```julia
firm_index = Dict(id => i for (i, id) in enumerate(unique(table.firm_id)))
f = [firm_index[id] for id in table.firm_id]
```

## One-Step API

```julia
using BipartiteGMRF

result = fit_mle(BipartiteNormalizedModel, f, w, y;
    solver=ExactCholesky(),
    seed=42,
)
```

## Two-Step API

Precompute sufficient statistics once and reuse them across fits:

```julia
ss = suffstats(BipartiteNormalizedModel, f, w, y;
    weighting=Weighting(observations=:raw),
)

result = fit_mle(BipartiteNormalizedModel, ss;
    solver=ExactCholesky(),
    seed=42,
)
```

To fit a custom-built model (for example with a hand-picked `rho_limit`):

```julia
model = BipartiteNormalizedModel(ss.A_prior; rho_limit=0.9)
result = fit_mle(model, ss; solver=ExactCholesky())
```

## Grouped AR(1) with iterative fitting

This small API example uses a co-managed first interval and one graph-only
manager link. All controls agree within each match. The short fit is not a
parameter-recovery or convergence claim.

```julia
using BipartiteGMRF, SparseArrays

f = [1, 1, 1, 1, 2, 2, 2, 3, 2]
w = [1, 2, 2, 3, 3, 4, 5, 5, 6]
y = [0.2, 0.2, 0.8, -0.3, 0.4, 1.1, -0.5, 0.7, NaN]
mid = [11, 11, 12, 13, 21, 22, 23, 31, 99]
rank = [1, 1, 2, 3, 1, 2, 3, 1, 0] # graph-only rank is ignored
X = sparse(hcat(ones(9), [-1, -1, 0, 1, -1, 0, 1, 0, 0]))
ss = suffstats(BipartiteVarianceStableModel, f, w, y;
    match_id=mid, edge_index=rank, error_eta=:estimate, X=X,
    weighting=Weighting(observations=:raw), standardize=false)
model = BipartiteVarianceStableModel(ss.A_prior; rho_limit=0.8)
solver = HutchSLQ(logdet_probes=64, lanczos_iters=20,
    cg_tol=1e-10, cg_maxiter=300, optim_iters=5)
result = fit_mle(model, ss; solver, seed=42, init=(eta=0.2,))
result.eta, result.beta, result.nll, result.converged

# Fixed eta: construct ss with error_eta=0.2 instead.
# Fixed rho: pass fix_rho=0.3 to fit_mle.
# Reuse the same seed when comparing starts or profiling rho.
```

`covariance(result)` would introduce a network factorization; it is optional
and not called above. `decompose(result)` currently rejects AR(1) results.

## StatsAPI Accessors

```julia
using StatsAPI

coef(result)          # [rho, sigma_a, sigma_z, sigma_epsilon]
coefnames(result)     # ["rho", "sigma_a", "sigma_z", "sigma_epsilon"]
params(result)        # (rho=..., sigma_a=..., sigma_z=..., sigma_epsilon=..., rho_eps=...)
loglikelihood(result) # original-units log-likelihood, constants included
aic(result); bic(result); dof(result); nobs(result)
```

## Variance Decomposition

```julia
model_vd  = decompose(result; kind=:model,  probes=100, seed=42)
fitted_vd = decompose(result; kind=:fitted, probes=100, seed=42)
```

## Effective Weighting

```julia
result = fit_mle(BipartiteNormalizedModel, f, w, y;
    weighting=Weighting(observations=:effective, rho_eps=:estimate),
    solver=ExactCholesky(),
)
result.rho_eps        # estimated within-match residual correlation
```

## Simulation

```julia
using SparseArrays, Random

A = sparse([1, 1, 2, 2, 3], [1, 2, 2, 3, 3], ones(5), 3, 3)
model = BipartiteNormalizedModel(A)
sim = simulate(model, A; ρ=0.5, σ_a=1.0, σ_z=0.8, σ_ε=0.3,
               rng=MersenneTwister(42))
```
