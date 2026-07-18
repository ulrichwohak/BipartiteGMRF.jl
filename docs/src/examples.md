# Examples

## One-Step API

```julia
using BipartiteGMRF, DataFrames

result = fit_mle(BipartiteNormalizedModel, df;
    outcome=:y,
    firm_id=:firm_id,
    worker_id=:worker_id,
    solver=ExactCholesky(),
    decompose=100,
    seed=42,
)
```

## Two-Step API

Precompute sufficient statistics once and reuse them across fits:

```julia
ss = suffstats(BipartiteNormalizedModel, df;
    outcome=:y,
    firm_id=:firm_id,
    worker_id=:worker_id,
    weighting=Weighting(observations=:raw),
)

result = fit_mle(BipartiteNormalizedModel, ss;
    solver=ExactCholesky(),
    decompose=100,
    seed=42,
)
```

To fit a custom-built model (for example with a hand-picked `rho_limit`):

```julia
model = BipartiteNormalizedModel(ss.A_prior; rho_limit=0.9)
result = fit_mle(model, ss; solver=ExactCholesky())
```

## StatsAPI Accessors

```julia
using StatsAPI

coef(result)          # [rho, sigma_a, sigma_z, sigma_epsilon]
coefnames(result)     # ["rho", "sigma_a", "sigma_z", "sigma_epsilon"]
params(result)        # (rho=..., sigma_a=..., sigma_z=..., sigma_epsilon=..., rho_eps=...)
loglikelihood(result) # original-units log-likelihood, constants included
aic(result); bic(result); dof(result); nobs(result)
```

## Effective Weighting

```julia
result = fit_mle(BipartiteNormalizedModel, df;
    weighting=Weighting(observations=:effective, rho_eps=:estimate),
    solver=ExactCholesky(),
    decompose=100,
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
