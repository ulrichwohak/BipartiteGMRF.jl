# Examples

## Two-Step API

```julia
using BipartiteGMRF, DataFrames

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

## Effective Weighting

```julia
result = gmrf_mle(
    df;
    weighting=Weighting(observations=:effective, rho_eps=:estimate),
    solver=ExactCholesky(),
    decompose=100,
)
```
